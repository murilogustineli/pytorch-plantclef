import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from .serde import deserialize_image
from matplotlib.font_manager import FontProperties

bold_font = FontProperties(weight="bold")


def crop_image_square(image: Image.Image) -> np.ndarray:
    min_dim = min(image.size)  # get the smallest dimension
    width, height = image.size
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    image = image.crop((left, top, right, bottom))
    image_array = np.array(image)
    return image_array


def plot_images_from_binary(
    df: pd.DataFrame,
    data_col: str,
    label_col: str,
    grid_size=(3, 3),
    crop_square: bool = False,
    figsize: tuple = (12, 12),
    dpi: int = 100,
):
    """
    Display images in a grid with binomial names as labels.

    :param df: DataFrame with the embeddings data.
    :param data_col: Name of the data column.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param crop_square: Boolean, whether to crop images to a square format by taking the center.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # collect binary image data from DataFrame
    subset_df = df.head(rows * cols)
    image_data_list = subset_df[data_col].tolist()
    image_names = subset_df[label_col].tolist()

    # create a matplotlib subplot with the specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # flatten the axes array for easy iteration if it's 2D
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, binary_data, name in zip(axes, image_data_list, image_names):
        # convert binary data to an image and display it
        image = deserialize_image(binary_data)

        # crop image to square if required
        if crop_square:
            image = crop_image_square(image)

        ax.imshow(image)
        name = name.replace("_", " ")
        wrapped_name = "\n".join(textwrap.wrap(name, width=25))
        ax.set_title(wrapped_name, fontsize=16, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        spines = ["top", "right", "bottom", "left"]
        for s in spines:
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_embeddings(
    df: pd.DataFrame,
    data_col: str,
    label_col: str,
    grid_size: tuple = (3, 3),
    figsize: tuple = (12, 12),
    dpi: int = 100,
):
    """
    Display images in a grid with species names as labels.

    :param df: DataFrame with the embeddings data.
    :param data_col: Name of the data column.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from DataFrame
    subset_df = df.head(rows * cols)
    embedding_data_list = subset_df[data_col].tolist()
    image_names = subset_df[label_col].tolist()

    # Create a matplotlib subplot with specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Flatten the axes array for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, embedding, name in zip(axes, embedding_data_list, image_names):
        embedding = np.array(embedding).flatten()

        # Find the next perfect square size greater than or equal to the embedding length
        next_square = math.ceil(math.sqrt(len(embedding))) ** 2
        padding_size = next_square - len(embedding)

        # Pad the embedding if necessary
        if padding_size > 0:
            embedding = np.pad(
                embedding, (0, padding_size), "constant", constant_values=0
            )

        # Reshape the embedding to a square
        side_length = int(math.sqrt(len(embedding)))
        # image_array = np.reshape(embedding, (side_length, side_length))
        image_array = embedding.reshape(side_length, side_length)

        # Normalize the embedding to [0, 255] for displaying as an image
        normalized_image = (
            (image_array - np.min(image_array))
            / (np.max(image_array) - np.min(image_array))
            * 255
        )
        image = Image.fromarray(normalized_image).convert("L")

        ax.imshow(image, cmap="gray")
        ax.set_xlabel(name)  # Set the species name as xlabel
        ax.xaxis.label.set_size(14)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def split_into_grid(image_array, grid_size: int = 3) -> list:
    h, w, c = image_array.shape  # Extract height, width, and channels
    grid_h, grid_w = h // grid_size, w // grid_size  # Tile size

    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            top, left = i * grid_h, j * grid_w
            bottom, right = top + grid_h, left + grid_w
            tile = image_array[top:bottom, left:right, :]  # Slice the NumPy array
            tiles.append(tile)

    return tiles  # Returns a list of NumPy arrays


def plot_image_tiles(
    df: pd.DataFrame,
    data_col: str = "data",
    grid_size: int = 3,
    figsize: tuple = (15, 8),
    dpi: int = 100,
):
    """
    Display an original image and its tiles in a single figure.

    :param df: DataFrame with the image data.
    :param data_col: Name of the data column containing image bytes.
    :param grid_size: Number of tiles per row/column (grid_size x grid_size).
    :param figsize: Figure size (width, height).
    :param dpi: Dots per inch (image resolution).
    """
    # extract the first row from DataFrame
    subset_df = df.head(1)
    image_data = subset_df[data_col].iloc[0]
    image_name = subset_df["image_name"].values[0]

    # convert binary image to PIL Image
    image = deserialize_image(image_data)
    image = crop_image_square(image)  # Ensure a square crop

    # split image into tiles
    image_tiles = split_into_grid(image, grid_size)

    # create figure with 1 row and 2 columns (original image | 3x3 grid)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Set main titles for left and right plots
    fig.text(0.25, 0.97, "Image", fontsize=20, fontweight="bold", ha="center")
    fig.text(0.75, 0.97, "Image Tiles", fontsize=20, fontweight="bold", ha="center")

    # plot original image on the left
    axes_left = axes[0]
    axes_left.imshow(image)
    axes_left.set_title(image_name, fontsize=20)
    axes_left.set_xticks([])
    axes_left.set_yticks([])
    spines = ["top", "right", "bottom", "left"]
    for spine in spines:
        axes_left.spines[spine].set_visible(False)

    # plot the grid of tiles on the right
    gs = fig.add_gridspec(1, 2)[1]
    grid_axes = gs.subgridspec(grid_size, grid_size)

    for idx, tile in enumerate(image_tiles):
        row, col = divmod(idx, grid_size)
        ax = fig.add_subplot(grid_axes[row, col])
        ax.imshow(tile)
        ax.set_xlabel(f"Tile {idx+1}", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in spines:
            ax.spines[s].set_visible(False)

    ax_right = axes[1]
    ax_right.set_title(f"{grid_size}x{grid_size} grid of tiles", fontsize=20)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def embeddings_to_image(embedding_list: list[np.array]) -> list[Image.Image]:
    embedding_images = []
    for embedding in embedding_list:
        # Find the next perfect square size greater than or equal to the embedding length
        next_square = math.ceil(math.sqrt(len(embedding))) ** 2
        padding_size = next_square - len(embedding)

        # Pad the embedding if necessary
        if padding_size > 0:
            embedding = np.pad(
                embedding, (0, padding_size), "constant", constant_values=0
            )

        # Reshape the embedding to a square
        side_length = int(math.sqrt(len(embedding)))
        image_array = np.reshape(embedding, (side_length, side_length))

        # Normalize the embedding to [0, 255] for displaying as an image
        normalized_image = (
            (image_array - np.min(image_array))
            / (np.max(image_array) - np.min(image_array))
            * 255
        )
        image = Image.fromarray(normalized_image).convert("L")
        embedding_images.append(image)

    return embedding_images


def plot_embed_tiles(
    df: pd.DataFrame,
    data_col: str,
    grid_size: int = 3,
    figsize: tuple = (15, 8),
    dpi: int = 100,
):
    """
    Display an original image and its tiles in a single figure.

    :param df: DataFrame with the image data.
    :param data_col: Name of the data column containing image bytes.
    :param grid_size: Number of tiles per row/column (grid_size x grid_size).
    :param figsize: Figure size (width, height).
    :param dpi: Dots per inch (image resolution).
    """
    # extract the first row from DataFrame
    subset_df = df.head(1)
    image_data = subset_df["data"].iloc[0]
    image_name = subset_df["image_name"].values[0]

    # convert binary image to PIL Image
    image = deserialize_image(image_data)
    image = crop_image_square(image)  # Ensure a square crop

    # get the embeddings from the DataFrame
    embed_data = df[data_col].iloc[0 : grid_size**2].tolist()
    embed_images = embeddings_to_image(embed_data)

    # create figure with 1 row and 2 columns (original image | 3x3 grid)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Set main titles for left and right plots
    fig.text(0.25, 0.97, "Image", fontsize=20, fontweight="bold", ha="center")
    fig.text(0.75, 0.97, "Tile Embeddings", fontsize=20, fontweight="bold", ha="center")

    # plot original image on the left
    axes_left = axes[0]
    axes_left.imshow(image)
    wrapped_name = "\n".join(textwrap.wrap(image_name, width=45))
    axes_left.set_title(wrapped_name, fontsize=20)
    axes_left.set_xticks([])
    axes_left.set_yticks([])
    spines = ["top", "right", "bottom", "left"]
    for spine in spines:
        axes_left.spines[spine].set_visible(False)

    # plot the grid of tiles on the right
    gs = fig.add_gridspec(1, 2)[1]
    grid_axes = gs.subgridspec(grid_size, grid_size)

    for idx, tile in enumerate(embed_images):
        row, col = divmod(idx, grid_size)
        ax = fig.add_subplot(grid_axes[row, col])
        ax.imshow(tile, cmap="gray")
        ax.set_xlabel(f"Tile {idx+1}", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in spines:
            ax.spines[s].set_visible(False)

    ax_right = axes[1]
    ax_right.set_title(f"{grid_size}x{grid_size} grid of tiles", fontsize=20)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_single_image_embeddings(
    df: pd.DataFrame,
    num_images: int = 2,
    figsize: tuple = (10, 12),
    dpi: int = 100,
):
    """
    Display an original image and its tiles in a single figure.

    :param df: DataFrame with the image data.
    :param data_col: Name of the data column containing image bytes.
    :param num_images: Number of single-label images to plot.
    :param figsize: Figure size (width, height).
    :param dpi: Dots per inch (image resolution).
    """
    # extract the first row from DataFrame
    image_data = df["data"].iloc[0:num_images].to_list()
    image_names = df["image_name"].iloc[0:num_images].to_list()
    embed_data = df["embeddings"].iloc[0:num_images].to_list()

    # Convert binary image to PIL Image
    images = [crop_image_square(deserialize_image(img)) for img in image_data]

    # Convert embeddings to images
    embed_images = embeddings_to_image(embed_data)

    # Create figure with (num_images x 2) grid (original | embeddings)
    fig, axes = plt.subplots(num_images, 2, figsize=figsize, dpi=dpi)

    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable when only one image

    # Set main titles for left and right plots
    fig.text(0.25, 0.95, "Images", fontsize=18, fontweight="bold", ha="center")
    fig.text(0.75, 0.95, "Embeddings", fontsize=18, fontweight="bold", ha="center")

    # Loop over images and embeddings
    for i, (img, emb_img, name) in enumerate(zip(images, embed_images, image_names)):
        # Left: Original image
        axes[i][0].imshow(img)
        wrapped_name = "\n".join(textwrap.wrap(name, width=25))
        axes[i][0].set_title(wrapped_name, fontsize=15)
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
        for spine in ["top", "right", "bottom", "left"]:
            axes[i][0].spines[spine].set_visible(False)

        # Right: Embedding visualization (grid)
        axes[i][1].imshow(emb_img, cmap="gray")
        axes[i][1].set_title(wrapped_name, fontsize=15)
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_faiss_classifications(
    embs_df: pd.DataFrame,
    faiss_df: pd.DataFrame,
    grid_size: int = 3,
    figsize: tuple = (20, 30),
    dpi: int = 100,
):
    """
    Display image tiles and their corresponding classifications generated by FAISS.

    :param embs_df: DataFrame containing training image embeddings and data.
    :param faiss_df: DataFrame containing FAISS predictions and similarities.
    :param grid_size: Number of tiles per row/column (grid_size x grid_size).
    :param figsize: Figure size (width, height).
    :param dpi: Dots per inch (image resolution).
    """
    num_tiles = grid_size**2

    # Extract data from test DataFrame
    image_data = faiss_df["data"].iloc[0]
    image_name = faiss_df["image_name"].iloc[0]

    # Convert binary image to PIL Image
    image = crop_image_square(deserialize_image(image_data))
    image_tiles = split_into_grid(image, grid_size)

    # Extract FAISS predictions for the first `num_tiles` tiles
    faiss_preds = faiss_df["predictions"].iloc[0:num_tiles].to_list()

    # Retrieve corresponding images from embs_df based on FAISS predictions
    faiss_images = []
    species_ids = []
    for pred_list in faiss_preds:
        tile_matches = []
        species_matches = []
        for img_name in pred_list:  # Iterate over the k predictions for each tile
            match = embs_df[embs_df["image_name"] == img_name]
            species_matches.append(match["species_id"].iloc[0])
            if not match.empty:
                faiss_img = crop_image_square(deserialize_image(match.iloc[0]["data"]))
                tile_matches.append(faiss_img)
        faiss_images.append(tile_matches)
        species_ids.append(species_matches)

    # Create figure with (num_tiles x k) grid (original | top-k embeddings)
    fig, axes = plt.subplots(
        num_tiles, len(faiss_images[0]) + 1, figsize=figsize, dpi=dpi
    )

    if num_tiles == 1:
        axes = [axes]  # Ensure axes is iterable when only one image

    # Set main titles for left and right plots
    fig.text(
        0.25,
        0.95,
        f"Image Tiles: {image_name}",
        fontsize=18,
        fontweight="bold",
        ha="center",
    )
    fig.text(
        0.75,
        0.95,
        "Top-K FAISS Predictions",
        fontsize=18,
        fontweight="bold",
        ha="center",
    )

    # Loop over image tiles and corresponding FAISS matches
    for i, (tile, matches, species) in enumerate(
        zip(image_tiles, faiss_images, species_ids)
    ):
        # Left: Original tile
        axes[i][0].imshow(tile)
        axes[i][0].set_title(f"Tile {i+1}", fontsize=15)
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
        axes[i][0].spines[:].set_visible(False)

        # Right: FAISS retrieved images
        for j, (faiss_img, species_id) in enumerate(zip(matches, species)):
            axes[i][j + 1].imshow(faiss_img)
            axes[i][j + 1].set_title(f"Top-{j+1}, SpeciesID: {species_id}", fontsize=12)
            axes[i][j + 1].set_xticks([])
            axes[i][j + 1].set_yticks([])
            axes[i][j + 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
