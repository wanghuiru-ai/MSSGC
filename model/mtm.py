
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from maml.fusions.factory import factory
from transformers import PreTrainedModel, PreTrainedTokenizer

from maml.text.dataset import MyDataPoint as TextDataPoint, MySentence as TextSentence, MyDataset as TextDataset
from maml.image.dataset import MyDataPoint as ImageDataPoint, MyImage, MyDataset as ImageDataset
from maml.data.dataset import MyDataPoint as MMDataPoint, MyPair as MMPair, MyDataset as MMDataset


def _use_cache(module: nn.Module, data_points: List[MMDataPoint]):
    cached, freeze = True, True
    for data_point in data_points:
        cached = cached and data_point.embedding is not None
    for parameter in module.parameters():
        freeze = freeze and not parameter.requires_grad
    return cached and freeze


def _use_tsa_cache(module: nn.Module, data_points: List[TextDataPoint]):
    cached, freeze = True, True
    for data_point in data_points:
        cached = cached and data_point.embedding is not None
    for parameter in module.parameters():
        freeze = freeze and not parameter.requires_grad
    return cached and freeze


def _use_image_cache(module: nn.Module, data_points: List[ImageDataPoint]):
    cached, freeze = True, True
    for data_point in data_points:
        cached = cached and data_point.embedding is not None
    for parameter in module.parameters():
        freeze = freeze and not parameter.requires_grad
    return cached and freeze


class MTM(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            sentence_embedding: PreTrainedModel,
            image_embedding: nn.Module,
            hidden_size: int = 2816,
            hidden_layer: int = 2,
    ):
        super(MTM, self).__init__()
        self.tokenizer = tokenizer
        self.sentence_embedding = sentence_embedding
        self.sentence_embedding_length = sentence_embedding.config.hidden_size

        self.image_embedding = image_embedding
        self.image_embedding_length = 2048

        self.fusion_type = 'cat_mlp'
        self.output_dim = 128
        self.sentence_fusion_length = 128
        self.image_fusion_length = 128
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(self.sentence_embedding_length, self.sentence_fusion_length)
        self.linear3 = nn.Linear(self.image_embedding_length, self.image_fusion_length)

        self.classifier1 = nn.Linear(self.sentence_fusion_length, 2)
        self.classifier2 = nn.Linear(self.output_dim, 2)
        self.classifier3 = nn.Linear(self.image_fusion_length, 2)

        self.dimensions = {
            1: [self.hidden_layer],
            2: [self.hidden_layer, self.hidden_layer],
            3: [self.hidden_layer, self.hidden_layer, self.hidden_layer],
        }

        self.fusion = factory({
            'type': self.fusion_type,
            'input_dims': [self.sentence_fusion_length, self.image_fusion_length],
            'output_dim': self.output_dim,
            'dimensions': self.dimensions[self.hidden_layer]
        })

        self.device = device
        self.to(device)

    def _embed_mm_sentences(self, pairs: List[MMPair]):
        sentence_list = [pair.sentence for pair in pairs]
        if _use_cache(self.sentence_embedding, sentence_list):
            return

        bs = 4
        for i in range(0, len(sentence_list), bs):
            sentences = sentence_list[i:i + bs]
            texts = [sentence.text for sentence in sentences]
            inputs = self.tokenizer(
                texts,
                max_length=512, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
            output = self.sentence_embedding(**inputs, return_dict=True)
            embeddings = output.last_hidden_state[:, 0, :]
            for sentence, embedding in zip(sentences, embeddings):
                sentence.embedding = embedding

    def _embed_mm_images(self, pairs: List[MMPair], second_forward: bool, embed_regions=False):
        images = [pair.image for pair in pairs]
        if _use_cache(self.image_embedding, images):
            return

        embeddings = torch.stack([image.data for image in images]).to(self.device)
        embeddings = self.image_embedding.embed(embeddings) if not embed_regions \
            else self.image_embedding.embed_regions(embeddings)
        for image, embedding in zip(images, embeddings):
            image.embedding = embedding
            if second_forward is True:
                image.data = None

    def _fuse_mm_text(self, pairs: List[MMPair]):
        self._embed_mm_sentences(pairs)
        return torch.stack([pair.sentence.embedding for pair in pairs])

    def _fuse_mm_image(self, pairs: List[MMPair], second_forward: bool):
        self._embed_mm_images(pairs, second_forward)
        return torch.stack([pair.image.embedding for pair in pairs])

    def _fuse_mm_concat(self, pairs: List[MMPair], second_forward: bool):
        self._embed_mm_sentences(pairs)
        self._embed_mm_images(pairs, second_forward)
        # return torch.stack([torch.cat((self.linear1(pair.sentence.embedding),
        #                                self.linear3(pair.image.embedding))) for pair in pairs])
        return torch.stack([self.fusion([self.linear1(pair.sentence.embedding),
                                         self.linear3(pair.image.embedding)]) for pair in pairs])

    def forward_mm(self, pairs: List[MMPair], second_forward: bool):
        fusion = {
            'concat': self._fuse_mm_concat,
        }
        embeddings = fusion['concat'](pairs, second_forward)
        scores = self.classifier2(self.dropout2(embeddings))
        return scores

    def forward_loss_mm(self, pairs: List[MMPair], second_forward=False):
        scores = self.forward_mm(pairs, second_forward)
        flags = torch.tensor([pair.flag for pair in pairs]).to(self.device)
        loss = F.cross_entropy(scores, flags)
        return loss

    def evaluate_mm(self, dataset: MMDataset, batch_size: int = 64) -> Tuple[float, float, float, float]:
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        data_loader = DataLoader(dataset, batch_size, collate_fn=list)
        data_loader = tqdm(data_loader, unit='batch')

        true_flags, pred_flags = [], []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                scores = self.forward_mm(batch, second_forward=True)
                true_flags += [pair.flag for pair in batch]
                pred_flags += torch.argmax(scores, dim=1).tolist()

        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        f1_score = f1_score(true_flags, pred_flags)
        precision = precision_score(true_flags, pred_flags)
        recall = recall_score(true_flags, pred_flags)
        accuracy = accuracy_score(true_flags, pred_flags)

        return f1_score, precision, recall, accuracy

    def _embed_images(self, images: List[MyImage], second_forward: bool, embed_regions=False):
        if _use_image_cache(self.image_embedding, images):
            return

        embeddings = torch.stack([image.data for image in images]).to(self.device)
        # print('embeddings: ', embeddings.shape)
        embeddings = self.image_embedding.embed(embeddings) if not embed_regions \
            else self.image_embedding.embed_regions(embeddings)
        # if self.training:
        #     embeddings = embeddings.unsqueeze(0)  # zero
        # print('embeddings2: ', embeddings.shape)
        for image, embedding in zip(images, embeddings):
            image.embedding = embedding
            if second_forward is True:
                image.data = None

    def _fuse_image(self, images: List[MyImage], second_forward: bool):
        self._embed_images(images, second_forward)
        return torch.stack([image.embedding for image in images])

    def forward_image(self, images: List[MyImage], second_forward: bool):
        fusion = {
            'image': self._fuse_image,
        }
        embeddings = fusion['image'](images, second_forward)
        scores = self.classifier3(self.dropout3(self.linear3(embeddings)))
        return scores

    def forward_loss_image(self, images: List[MyImage], second_forward: bool = False):
        scores = self.forward_image(images, second_forward)
        flags = torch.tensor([image.flag for image in images]).to(self.device)
        loss = F.cross_entropy(scores, flags)
        return loss

    def evaluate_image(self, dataset: ImageDataset, batch_size: int = 64) -> Tuple[float, float, float, float]:
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        data_loader = DataLoader(dataset, batch_size, collate_fn=list)
        data_loader = tqdm(data_loader, unit='batch')

        true_flags, pred_flags = [], []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                scores = self.forward_image(batch, second_forward=True)
                true_flags += [pair.flag for pair in batch]
                # pred_flags += [0 if score < 0.5 else 1 for score in scores]
                pred_flags += torch.argmax(scores, dim=1).tolist()

        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        f1_score = f1_score(true_flags, pred_flags)
        precision = precision_score(true_flags, pred_flags)
        recall = recall_score(true_flags, pred_flags)
        accuracy = accuracy_score(true_flags, pred_flags)

        return f1_score, precision, recall, accuracy

    def _embed_sentences(self, sentence_list: List[TextSentence]):
        bs = 1

        for i in range(0, len(sentence_list), bs):
            sentences = sentence_list[i:i + bs]
            texts = [sentence.text for sentence in sentences]
            inputs = self.tokenizer(
                texts,
                max_length=512, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
            output = self.sentence_embedding(**inputs, return_dict=True)
            embeddings = output.last_hidden_state[:, 0, :]
            for sentence, embedding in zip(sentences, embeddings):
                sentence.embedding = embedding

    def _fuse_text(self, sentence_list: List[TextSentence]):
        self._embed_sentences(sentence_list)
        return torch.stack([sentence.embedding for sentence in sentence_list])

    def forward_text(self, sentence_list: List[TextSentence]):
        fusion = {
            'text': self._fuse_text,
        }
        embeddings = fusion['text'](sentence_list)
        scores = self.classifier1(self.dropout1(self.linear1(embeddings)))
        # print(scores.shape)
        return scores

    def forward_loss_text(self, sentence_list: List[TextSentence]):
        scores = self.forward_text(sentence_list)
        flags = torch.tensor([sentence.flag for sentence in sentence_list]).to(self.device)
        loss = F.cross_entropy(scores, flags)
        return loss

    def evaluate_text(self, dataset: TextDataset, batch_size: int = 64) -> Tuple[float, float, float]:
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        data_loader = DataLoader(dataset, batch_size, collate_fn=list)
        data_loader = tqdm(data_loader, unit='batch')

        true_flags, pred_flags = [], []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                scores = self.forward_text(batch)
                true_flags += [pair.flag for pair in batch]
                pred_flags += torch.argmax(scores, dim=1).tolist()

        from sklearn.metrics import f1_score, recall_score, accuracy_score
        recall = recall_score(true_flags, pred_flags, average='macro')
        f1_n, f1_p = f1_score(true_flags, pred_flags, average=None)
        f1_pn = (f1_p + f1_n) / 2
        accuracy = accuracy_score(true_flags, pred_flags)

        return recall, f1_pn, accuracy


