from .forward_pass import validate_network
from .generic import create_pytorch_objects
import os
from pathlib import Path

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import pickle, argparse, torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.io import imsave
from tqdm import tqdm
from torch.cuda.amp import autocast
import tiffslide as openslide
from GANDLF.data import get_testing_loader
from GANDLF.utils import (
    populate_channel_keys_in_params,
    send_model_to_device,
    get_dataframe,
    best_model_path_end,
    load_model,
)
from GANDLF.models import get_model
from GANDLF.data.inference_dataloader_histopath import InferTumorSegDataset


def inference_loop(
    inferenceDataFromPickle, device, parameters, outputDir_or_optimizedModel
):
    """
    The main training loop.

    Args:
        inferenceDataFromPickle (pandas.DataFrame): The data to use for inference.
        device (str): The device to perform computations on.
        parameters (dict): The parameters dictionary.
        outputDir_or_optimizedModel (str): The output directory or optimized model file.
    """
    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    print("Number of classes  : ", len(parameters["model"]["class_list"]))
    parameters["testing_data"] = inferenceDataFromPickle

    (
        model,
        _,
        _,
        _,
        _,
        parameters,
    ) = create_pytorch_objects(parameters, device=device)

    # Fetch the model according to params mentioned in the configuration file
    model = get_model(parameters)

    # Setting up the inference loader
    inference_loader = get_testing_loader(parameters)

    # Loading the weights into the model
    model_file = outputDir_or_optimizedModel
    if os.path.isdir(model_file):
        model_file = os.path.join(
            outputDir_or_optimizedModel,
            str(parameters["model"]["architecture"]) + best_model_path_end,
        )

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            "The model specified in file was not found:", model_file
        )

    main_dict = load_model(
        model_file, device=torch.device(device), full_sanity_check=False
    )
    model.load_state_dict(main_dict["model_state_dict"])

    if not (os.environ.get("HOSTNAME") is None):
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    # get the channel keys for concatenation later (exclude non numeric channel keys)
    parameters = populate_channel_keys_in_params(inference_loader, parameters)
    parameters["save_output"] = True

    print("Data Samples: ", len(inference_loader.dataset), flush=True)
    model, parameters["model"]["amp"], parameters["device"] = send_model_to_device(
        model, parameters["model"]["amp"], device, optimizer=None
    )

    print("Using device:", parameters["device"], flush=True)

    # radiology inference
    if parameters["modality"] == "rad":
        average_epoch_valid_loss, average_epoch_valid_metric = validate_network(
            model, inference_loader, None, parameters, mode="inference"
        )
        print(average_epoch_valid_loss, average_epoch_valid_metric)
    elif parameters["modality"] in ["path", "histo"]:
        # set some defaults
        if not "slide_level" in parameters:
            parameters["slide_level"] = 0
        if not "stride_size" in parameters:
            parameters["stride_size"] = parameters["patch_size"]

        parameters["stride_size"] = np.array(parameters["stride_size"])

        if parameters["stride_size"].size == 1:
            parameters["stride_size"] = np.append(
                parameters["stride_size"], parameters["stride_size"]
            )

        if not "mask_level" in parameters:
            parameters["mask_level"] = parameters["slide_level"]

        # actual computation
        for _, row in inferenceDataFromPickle.iterrows():
            subject_name = row[parameters["headers"]["subjectIDHeader"]]
            os_image = openslide.open_slide(
                row[parameters["headers"]["channelHeaders"]].values[0]
            )
            level_width, level_height = os_image.level_dimensions[
                int(parameters["slide_level"])
            ]
            subject_dest_dir = os.path.join(
                outputDir_or_optimizedModel, str(subject_name)
            )
            Path(subject_dest_dir).mkdir(parents=True, exist_ok=True)

            probs_map = np.zeros((level_height, level_width), dtype=np.float16)
            count_map = np.zeros((level_height, level_width), dtype=np.uint8)

            patch_size = parameters["patch_size"]

            patient_dataset_obj = InferTumorSegDataset(
                row[parameters["headers"]["channelHeaders"]].values[0],
                patch_size=patch_size,
                stride_size=parameters["stride_size"],
                selected_level=parameters["slide_level"],
                mask_level=parameters["mask_level"],
            )

            dataloader = DataLoader(
                patient_dataset_obj,
                batch_size=int(parameters["batch_size"]),
                shuffle=False,
                num_workers=parameters["q_num_workers"],
            )
            for image_patches, (x_coords, y_coords) in tqdm(dataloader):
                x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
                if parameters["model"]["amp"]:
                    with autocast():
                        output = model(image_patches.float().to(parameters["device"]))
                else:
                    output = model(image_patches.float().to(parameters["device"]))
                output = output.detach().cpu().numpy()
                for i in range(int(output.shape[0])):
                    count_map[
                        x_coords[i] : x_coords[i] + patch_size[0],
                        y_coords[i] : y_coords[i] + patch_size[1],
                    ] += 1
                    probs_map[
                        x_coords[i] : x_coords[i] + patch_size[0],
                        y_coords[i] : y_coords[i] + patch_size[1],
                    ] += output[i][0]
            probs_map = probs_map / count_map
            count_map = count_map / count_map.max()
            out = count_map * probs_map
            count_map = np.array(count_map * 255, dtype=np.uint16)
            out_thresh = np.array((out > 0.5) * 255, dtype=np.uint16)
            imsave(
                os.path.join(
                    subject_dest_dir,
                    str(row[parameters["headers"]["subjectIDHeader"]]) + "_prob.png",
                ),
                out,
            )
            imsave(
                os.path.join(
                    subject_dest_dir,
                    str(row[parameters["headers"]["subjectIDHeader"]]) + "_seg.png",
                ),
                out_thresh,
            )
            imsave(
                os.path.join(
                    subject_dest_dir,
                    str(row[parameters["headers"]["subjectIDHeader"]]) + "_count.png",
                ),
                count_map,
            )


if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description="Inference Loop of GANDLF")
    parser.add_argument(
        "-inference_loader_pickle",
        type=str,
        help="Inference loader pickle",
        required=True,
    )
    parser.add_argument(
        "-parameter_pickle", type=str, help="Parameters pickle", required=True
    )
    parser.add_argument(
        "-headers_pickle", type=str, help="Header pickle", required=True
    )
    parser.add_argument("-outputDir", type=str, help="Output directory", required=True)
    parser.add_argument("-device", type=str, help="Device to train on", required=True)

    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    patch_size = pickle.load(open(args.patch_size_pickle, "rb"))
    headers = pickle.load(open(args.headers_pickle, "rb"))
    label_header = pickle.load(open(args.label_header_pickle, "rb"))
    parameters = pickle.load(open(args.parameter_pickle, "rb"))
    inferenceDataFromPickle = get_dataframe(args.inference_loader_pickle)

    inference_loop(
        inferenceDataFromPickle=inferenceDataFromPickle,
        parameters=parameters,
        outputDir_or_optimizedModel=args.outputDir,
        device=args.device,
    )
