"""Linear-chain Conditional Random Field (CRF) layer for sequence labeling."""

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain CRF.

    Learns a (num_tags x num_tags) transition matrix so that decoding
    produces globally-optimal label sequences rather than per-position argmax.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] = score of transitioning from tag i -> tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: (batch, seq_len, num_tags) — unnormalized scores (logits).
            tags: (batch, seq_len) — ground-truth tag indices.
            mask: (batch, seq_len) — boolean, True for valid positions.

        Returns:
            Scalar mean NLL loss over the batch.
        """
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        # NLL = log Z(x) - score(x, y), normalized per token
        nll = forward_score - gold_score  # (batch,)
        lengths = mask.float().sum(dim=1)  # (batch,)
        nll = nll / lengths  # per-token NLL
        return nll.mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi decode to find the best tag sequence.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len) boolean

        Returns:
            List of lists — best tag index sequence per batch element.
        """
        return self._viterbi_decode(emissions, mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score of the gold tag sequence under the model."""
        batch, seq_len, _ = emissions.shape
        mask_f = mask.float()

        # Clamp tags so padding positions (-100) don't cause index errors
        safe_tags = tags.clamp(min=0)

        # Emission scores for gold tags
        emit_scores = emissions.gather(2, safe_tags.unsqueeze(2)).squeeze(2)  # (B, T)
        emit_scores = (emit_scores * mask_f).sum(dim=1)  # (B,)

        # Start transition
        score = self.start_transitions[safe_tags[:, 0]]  # (B,)
        score = score + emit_scores

        # Transition scores between consecutive tags
        for t in range(seq_len - 1):
            cur_tag = safe_tags[:, t]      # (B,)
            next_tag = safe_tags[:, t + 1]  # (B,)
            trans = self.transitions[cur_tag, next_tag]  # (B,)
            score = score + trans * mask_f[:, t + 1]

        # End transition at the last valid position
        lengths = mask_f.sum(dim=1).long()  # (B,)
        last_tags = safe_tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)  # (B,)
        score = score + self.end_transitions[last_tags]

        return score  # (B,)

    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-partition function (log Z) via the forward algorithm."""
        batch, seq_len, num_tags = emissions.shape
        mask_f = mask.float()

        # alpha[b, j] = log-sum-exp of all partial paths ending in tag j
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, T_tags)

        for t in range(1, seq_len):
            # alpha_t[b, j] = logsumexp_i(alpha[b, i] + transitions[i, j]) + emit[b, t, j]
            emit = emissions[:, t].unsqueeze(1)  # (B, 1, num_tags)
            trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            alpha_exp = alpha.unsqueeze(2)  # (B, num_tags, 1)
            scores = alpha_exp + trans + emit  # (B, num_tags, num_tags)
            new_alpha = torch.logsumexp(scores, dim=1)  # (B, num_tags)

            # Only update positions where mask is True
            m = mask_f[:, t].unsqueeze(1)  # (B, 1)
            alpha = new_alpha * m + alpha * (1 - m)

        # Add end transitions
        alpha = alpha + self.end_transitions.unsqueeze(0)  # (B, num_tags)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi algorithm for finding the best tag sequences."""
        batch, seq_len, num_tags = emissions.shape
        mask_f = mask.float()

        # viterbi[b, j] = best score of partial path ending in tag j
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, num_tags)
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            emit = emissions[:, t]  # (B, num_tags)
            trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            v_exp = viterbi.unsqueeze(2)  # (B, num_tags, 1)
            scores = v_exp + trans  # (B, num_tags, num_tags)
            best_scores, best_tags = scores.max(dim=1)  # (B, num_tags), (B, num_tags)
            new_viterbi = best_scores + emit

            m = mask_f[:, t].unsqueeze(1)
            viterbi = new_viterbi * m + viterbi * (1 - m)
            backpointers.append(best_tags)

        # Add end transitions
        viterbi = viterbi + self.end_transitions.unsqueeze(0)

        # Traceback
        lengths = mask_f.sum(dim=1).long()  # (B,)
        best_last_tags = viterbi.argmax(dim=1)  # (B,)

        result: List[List[int]] = []
        for b in range(batch):
            L = int(lengths[b].item())
            path = [0] * L
            path[-1] = int(best_last_tags[b].item())
            for t in range(L - 2, -1, -1):
                path[t] = int(backpointers[t][b, path[t + 1]].item())
            result.append(path)

        return result
