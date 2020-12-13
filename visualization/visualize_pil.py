import matplotlib.pyplot as plt


def display_pil_images(images: list,
                       masks: list = None,
                       labels: list = None,
                       columns: int = 5,
                       width: int = 20,
                       height: int = 8,
                       label_font_size: int = 9):
    """

    Plot multiple images (max 15) in a grid-like structure. Masks are applied over images.

    :param images: list of PIL images
    :param masks: list of PIL masks
    :param labels: image labels
    :param columns: number of columns to display
    :param width: plot's width
    :param height: plot's height
    :param label_font_size: label's font size
    """
    max_images = 15

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]
        if masks is not None:
            masks = masks[0:max_images]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))

    if masks is not None:
        for i, (image, mask) in enumerate(zip(images, masks)):
            plt.subplot(int(len(images) / columns) + 1, columns, i + 1)
            plt.imshow(image)
            plt.imshow(mask, cmap='coolwarm', alpha=0.5)

            if labels is not None:
                plt.title(labels[i], fontsize=label_font_size)
    else:
        for i, image in enumerate(images):
            plt.subplot(int(len(images) / columns) + 1, columns, i + 1)
            plt.imshow(image)

            if labels is not None:
                plt.title(labels[i], fontsize=label_font_size)
    plt.show()


def visualize(**images):
    """

    Plot PIL images in one row.

    :param images list of PIL images
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
