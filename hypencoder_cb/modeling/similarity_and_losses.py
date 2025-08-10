from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from hypencoder_cb.modeling.shared import EncoderOutput


def pos_neg_triplets_from_similarity(similarity: torch.Tensor) -> torch.Tensor:
    """Takes a similarity matrix and turns it into a matrix of
    positive-negative pairs.

    Args:
        similarity (torch.Tensor): A similarity matrix with shape:
                (num_queries, num_items_per_query).
            It is assumed that the first item in each row is the positive item.

    Returns:
        torch.Tensor: A matrix of positive-negative pairs with shape:
            (num_queries * num_negatives_per_query, 2).
    """

    num_queries, num_items_per_query = similarity.shape
    num_negatives_per_query = num_items_per_query - 1

    if num_items_per_query == 2:
        return similarity

    assert num_items_per_query > 2

    # Extract the scores for all the positive items.
    positives = similarity[:, 0]

    output = torch.zeros(
        num_queries * num_negatives_per_query, 2, device=similarity.device
    )
    # Filling the first column with the positive scores by repeating each
    # positive score N times. N is the number of negative items per query.
    output[:, 0] = positives.repeat_interleave(num_negatives_per_query)

    # Filling the second column with the negative scores by placing the 
    # scores of the query's negative items next to the repeated positive
    # score of this query.
    for i in range(num_queries):
        output[
            i * num_negatives_per_query : (i + 1) * num_negatives_per_query, 1
        ] = similarity[i, 1:]

    return output


def no_in_batch_negatives_hypecoder_similarity(
    query_models: Callable,
    item_embeddings: torch.Tensor,
    required_num_items_per_query: Optional[int] = None,
) -> torch.Tensor:
    """Takes a set of query models and a set of item embeddings and returns
    the similarity between each query and each item.

    Args:
        query_models (Callable): A callable that takes a tensor of items and
            returns a tensor of similarities.
        item_embeddings (torch.Tensor): A tensor of item embeddings (document
         embeddings) with shape: (num_items, item_emb_dim).
        required_num_items_per_query (Optional[int], optional): An optional
            integer that specifies the number of items required per query.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of similarities with shape:
            (num_queries, num_items_per_query).
    """

    assert len(item_embeddings.shape) == 2

    num_items, item_emb_dim = item_embeddings.shape
    num_queries = query_models.num_queries

    # Validates that the number of items/documents is multiple of the 
    # number of queries.
    assert num_items % num_queries == 0

    num_items_per_query = num_items // num_queries

    if required_num_items_per_query is not None:
        assert num_items_per_query == required_num_items_per_query

    # Reshape the item embeddings into a structured batch, where each
    # row corresponds to a query. 
    item_embeddings = item_embeddings.view(
        num_queries, num_items_per_query, item_emb_dim
    )

    # calls the q-net (query model) passing the structured batch of
    # document embeddings. The q-net applies the query-specific scoring
    # function to each document embedding in the batch.
    similarity = query_models(item_embeddings).squeeze()

    return similarity


def in_batch_negatives_hypecoder_similarity(
    query_models: Callable,
    item_embeddings: torch.Tensor,
    required_num_items_per_query: Optional[int] = None,
) -> torch.Tensor:
    """Takes a set of query models and a set of item embeddings and returns
    the similarity between each query and each item.

    Args:
        query_models (Callable): A callable that takes a tensor of items and
            returns a tensor of similarities.
        item_embeddings (torch.Tensor): A tensor of item embeddings with shape:
            (num_items, item_emb_dim).
        required_num_items_per_query (Optional[int], optional): An optional
            integer that specifies the number of items required per query.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of similarities with shape:
            (num_queries, num_items).
    """

    assert len(item_embeddings.shape) == 2

    num_items, item_emb_dim = item_embeddings.shape
    num_queries = query_models.num_queries

    # Repeat the item embeddings for each query the new shape is:
    # (num_queries, num_items, item_emb_dim)
    item_embeddings = item_embeddings.unsqueeze(0).repeat(num_queries, 1, 1)

    # Passes the expanded tensor to the q-net. 
    # The q-net is batched by num_queries. 
    # The q-net applies the query-specific scoring function to each
    # item/document in the batch
    similarity = (
        query_models(item_embeddings).view(num_queries, num_items).squeeze()
    )

    return similarity


@dataclass
class SimilarityAndLossOutput:
    # Provides a standardized return type for all loss functions
    similarity: torch.Tensor
    loss: torch.Tensor


class SimilarityAndLossBase(nn.Module):
    # Abstract base class for defining the interface for all loss functions
    # It establishes the pattern of separating similarity calculations
    # from loss calculations.

    def __init__(self, *args, scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    # Abstract method for calculating similarity
    # Subclasses must define the implementation
    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    # Abstract method for calculating the loss from the similarity score
    # Subclasses must define the implementation
    def _loss(
        self, similarity: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    # The main entry point. It orchestrates the process by first calling
    # self._get_similarity() and then passing the result to self._loss().
    # It returns a SimilarityAndLossOutput object.
    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SimilarityAndLossOutput:
        
        similarity = self._get_similarity(query_output, passage_output, **kwargs)

        loss = self.scale * self._loss(similarity, labels, **kwargs)

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class MarginMSELoss(SimilarityAndLossBase):
    # Generic implementation of the MarginMSE knowledge distillation loss
    # It implements the SimilarityAndLossBase abstract class

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.MSELoss()

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _loss(
        self, similarity: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            similarity: similarity scores predicted by the student model
                        shape (positive_score, negative_score)
            labels: similarity scores predicted by the teacher model
                    shape (positive_score, negative_score)
        """
        num_similarity_queries, num_similarity_items = similarity.shape
        num_label_queries, num_label_items = labels.shape

        assert num_similarity_items == 2
        assert num_label_items > 1

        if num_label_items != 2:
            labels = pos_neg_triplets_from_similarity(labels)
            num_label_queries, num_label_items = labels.shape

        assert num_label_items == 2
        assert num_similarity_queries == num_label_queries

        # --- THESE ARE THE LINES TO REMOVE ---
        # normalization_fn() method does not exit anywhere in this 
        # codebase or in pytorch or huggingface 
        # similarity = self.normalization_fn(similarity)
        # labels = self.normalization_fn(labels)
        # -------------------------------------
        # Calculating the margin score from student model predictions
        margin = similarity[:, 0] - similarity[:, 1]
        # Calculating the margin score from teacher model
        teacher_margin = labels[:, 0] - labels[:, 1]

        # Calculates the MSE between the student's margin score and 
        # the teacher margin score
        return self.loss(margin.squeeze(), teacher_margin.squeeze())


class CrossEntropyLoss(SimilarityAndLossBase):

    # Generic implementation of a contrastive, cross-entropy-based loss.
    def __init__(
        self,
        use_in_batch_negatives: bool = True,
        only_use_first_item: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = nn.CrossEntropyLoss()

        self.use_in_batch_negatives = use_in_batch_negatives
        self.only_use_first_item = only_use_first_item

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _loss(
        self, similarity: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.loss(similarity, labels)

    # It creates the ground truth labels for the CE loss.
    def _get_target(
        self,
        num_queries: int,
        num_items: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_items_per_query = num_items // num_queries
        
        if self.use_in_batch_negatives:
            # the correct "class" for query i is the positive document
            # at index i * items_per_query
            # Creates a tensor like [0, 1, 2, 3, ...] and scales it.
            targets = torch.arange(
                num_queries,
                dtype=torch.long,
                device=device,
            )

            # If items_per_query is 1, targets = [0, 1].
            # If we had 2 positives per query, targets would be [0, 2]
            targets = targets * num_items_per_query

        else:
            # the correct class is at index 0.
            targets = torch.zeros(num_queries, dtype=torch.long, device=device)

        return targets

    # Overrides the parent class implementation because it needs to 
    # create its target labels
    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SimilarityAndLossOutput:
        similarity = self._get_similarity(
            query_output, passage_output, **kwargs
        )

        target = self._get_target(
            similarity.size(0), similarity.size(1), device=similarity.device
        )

        loss = self.scale * self._loss(similarity, target, **kwargs)

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class HypencoderMarginMSELoss(MarginMSELoss):
    # This inherits the loss calculation logic from the parent class
    # and provide its implementation of the _get_similarity method
    
    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        # The class's implementation of _get_similarity().


        # It passes the callable q-net and the doc embeddings to the
        # bellow method
        similarity = no_in_batch_negatives_hypecoder_similarity(
            query_output.representation,
            passage_output.representation,
        )

        # Formatting the output properly for the parent's _loss method
        return pos_neg_triplets_from_similarity(similarity)

    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
    ) -> SimilarityAndLossOutput:
        loss = torch.tensor(0.0, device=passage_output.representation.device)
        similarity = self._get_similarity(query_output, passage_output)
        loss += self.scale * self._loss(
            similarity,
            labels,
        )

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class HypencoderCrossEntropyLoss(CrossEntropyLoss):
    # The concrete implementation of CE loss for the Hypencoder.
    def __init__(
        self,
        use_query_embedding_representation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_query_embedding_representation = (
            use_query_embedding_representation
        )

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_in_batch_negatives:
            if self.use_cross_device_negatives:
                raise NotImplementedError(
                    "Cross device negatives not supported for Hypencoder."
                )
            else:
                query_model = query_output.representation
                passage_embeddings = passage_output.representation

            if self.only_use_first_item:
                num_items = passage_embeddings.shape[0]
                num_queries = query_model.num_queries
                items_per_query = num_items // num_queries

                indices = (
                    torch.arange(
                        num_queries,
                        device=passage_embeddings.device,
                        dtype=torch.long,
                    )
                    * items_per_query
                )

                passage_embeddings = passage_embeddings[indices]

            similarity = in_batch_negatives_hypecoder_similarity(
                query_model, passage_embeddings
            )
        else:
            similarity = no_in_batch_negatives_hypecoder_similarity(
                query_output.representation, passage_output.representation
            )

        return similarity
