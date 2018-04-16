

from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import torch
import json
from PIL import Image
import numpy as np
import os
from string import punctuation


punc = str.maketrans(dict.fromkeys(punctuation))


class CLEVRDataset(Dataset):
    def __init__(self, root, download=True, mode='train', transform=None):
        # TODO
        # 1. Initialize file path or list of file names.

        questions_file = root + 'questions/CLEVR_{}_questions.json'.format(mode)
        self.image_folder = root + 'images/{}/'.format(mode)
        self.mode = mode

        self.questions = json.load(open(questions_file))['questions']

        with open('data/words.pkl', 'rb') as f:
            self.words_to_indices, self.indices_to_words, self.answer_to_indices, self.indices_to_answers = pickle.load(f)

        with open('data/data_{}.pkl'.format(mode), 'rb') as f:
            if mode != 'test':
                self.images, self.questions, self.answers = pickle.load(f)
                # self.images, self.questions, self.answers = self.images[:100], self.questions[:100], self.answers[:100]
            else:
                self.images, self.questions = pickle.load(f)
                # self.images, self.questions = self.images[:100], self.questions[:100]

        self.transform = transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        if self.mode != 'test':
            image_file, question, answer = self.images[index], self.questions[index], self.answers[index]
        else:
            image_file, question = self.images[index], self.questions[index]

        image_file = self.image_folder + image_file

        image = Image.open(image_file)
        image = np.array(image)[:, :, :3]

        if self.transform is not None:
            image = self.transform(image)

        question = torch.LongTensor(question)

        if self.mode != 'test':
            answer = torch.LongTensor([answer-1])
            return image, question, answer
        else:
            return image, question

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.questions)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, questions, answers = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(q) for q in questions]
    targets = torch.zeros(len(questions), max(lengths)).long()
    for i, q in enumerate(questions):
        end = lengths[i]
        targets[i, :end] = q[:end]

    answers = torch.stack(answers)
    lengths = torch.LongTensor(lengths) - 1
    return images, targets, lengths, answers


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4641, 0.4605, 0.4542],
                         std=[0.0114, 0.0114, 0.0127])
])


if __name__ == '__main__':

    root = 'data/CLEVR_v1.0/'
    # CLEVRDataset(root, download=False, mode='train')
    # CLEVRDataset(root, download=False, mode='val')
    CLEVRDataset(root, download=False, mode='test')
