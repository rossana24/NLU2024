from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
import torch


def distant_cross_entropy(logits, positions, mask=None):
    """
    Compute the distant cross-entropy loss for span extraction.
    :param logits: Tensor of shape [N, L, 2] where N=batch size, L=sequence length, and 2 represents the start and end logits.
    :param positions: Tensor of shape [N, L] indicating the ground truth span positions (1 for correct position, 0 otherwise).
    :param mask: Tensor of shape [N] to mask out invalid positions (e.g., padding tokens). Default is None.
    :return: Scalar tensor representing the computed loss.
    """
    log_softmax = nn.LogSoftmax(dim=-1)  # LogSoftmax function to apply to logits
    log_probs = log_softmax(logits)  # Compute log probabilities from logits

    # Compute loss
    if mask is not None:
        # Compute weighted loss if mask is provided
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(
                                   dtype=log_probs.dtype)))
    else:
        # Compute unweighted loss if no mask is provided
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss


class BertForSpanAspectExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSpanAspectExtraction, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        """
        Forward pass of the model.

        :param input_ids: Tensor of shape [N, L] with token IDs for the input sequence.
        :param token_type_ids: Tensor of shape [N, L] with segment IDs to differentiate between sentences.
        :param attention_mask: Tensor of shape [N, L] indicating which tokens should be attended to (1) or masked (0).
        :param start_positions: Optional tensor of shape [N] with ground truth start positions for training.
        :param end_positions: Optional tensor of shape [N] with ground truth end positions for training.
        :return: A dictionary containing the computed losses and logits for start and end positions.
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Extract the last hidden state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # Remove the last dimension (1) to get shape [N, L]
        end_logits = end_logits.squeeze(-1)  # Remove the last dimension (1) to get shape [N, L]

        # Compute losses if ground truth positions are provided
        if start_positions is not None and end_positions is not None:
            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2  # Average the start and end losses
        else:
            total_loss = None  # No loss computation if ground truth positions are not provided

        return {"total_loss": total_loss,
                "start_loss": start_loss,
                "end_loss": end_loss,
                "start_positions": start_logits,
                "end_positions": end_logits}
