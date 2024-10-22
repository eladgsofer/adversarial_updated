import torch
import clip

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_path


class Tagger:
    def __init__(self, tag_dict):

        self.tag_dict = {tag_id: ["The photo {0} is {1}".format(tag_id, label) for label in tag_values]
                         for tag_id, tag_values in tag_dict.items()}

        self.model, self.preprocess = clip.load("ViT-B/32")  # Load CLIP model

        self.tag_dict = {tag_id: self.model.encode_text(clip.tokenize(tag_values))
                         for tag_id, tag_values in tag_dict.items()}

    def tag_images(self, dataloader):
        all_tags = {}
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for images, image_paths in dataloader:
                # Encode the batch of images
                image_features = self.model.encode_image(images)

                # Iterate through each tag category
                for category, tags_text_embeddings in self.tag_dict.items():

                    # Compute similarities between image and text features
                    similarities = torch.cosine_similarity(image_features.unsqueeze(1),
                                                           tags_text_embeddings.unsqueeze(0), dim=2)

                    # For each image in the batch, select the most similar tag for each category
                    for i, image_path in enumerate(image_paths):
                        max_similarity, max_index = torch.max(similarities[i], dim=0)
                        if max_similarity > 0.5:  # Threshold for relevant tags
                            if image_path not in all_tags:
                                all_tags[image_path] = {}
                            all_tags[image_path][category] = tags[max_index]

        return all_tags


from torch.utils.data import DataLoader

# Example image paths and tag dictionary

# The
image_paths = ["path_to_image1.jpg", "path_to_image2.jpg"]
tag_dict = {'location': ['indoor', 'outdoor'], 'dominant_color': ['blue', 'red', 'green']}


# Initialize Dataset and DataLoader
dataset = ImageDataset(image_paths, transform=tagger.preprocess)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Create Tagger instance and process images
tagger = Tagger(tag_dict)
tagged_images = tagger.tag_images(dataloader)
print(tagged_images)