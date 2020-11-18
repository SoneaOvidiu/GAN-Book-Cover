from os import mkdir, listdir
from os.path import exists, join
from remove_text_from_image import remove_text_from_image


input_images_directory = "book-covers"
output_images_directory = "book-covers-no-text"
discarded_images_directory = "book-covers-discarded"
categories = listdir(input_images_directory)

if not exists(output_images_directory):
    mkdir(output_images_directory)
if not exists(discarded_images_directory):
    mkdir(discarded_images_directory)

for category_directory in categories:
    input_category_path = join(input_images_directory, category_directory)
    output_category_path = join(output_images_directory, category_directory)
    discarded_category_path = join(discarded_images_directory, category_directory)

    if not exists(output_category_path):
        mkdir(output_category_path)
    if not exists(discarded_category_path):
        mkdir(discarded_category_path)

    for input_file in listdir(input_category_path):
        if not input_file.endswith(".csv"):
            input_image = input_file
            input_image_path = join(input_category_path, input_image)
            output_image_path = join(output_category_path, input_image)
            discarded_image_path = join(discarded_category_path, input_image)

            remove_text_from_image(input_image_path, output_image_path, discarded_image_path)
