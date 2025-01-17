from typing import List, Tuple, Union
from src.interactants import UNSPECIFIC_ID
import torch


class DNALayout:
    def __init__(
            self,
            size: int,
            dna_spec: List[Tuple[int, int, int]],
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        self.device = device
        self.size = size
        self.dna = torch.zeros(self.size, dtype=torch.uint8).to(self.device)
        for start, end, seg in dna_spec:
            self.dna[start:end] = seg
        self.dna_spec_size = self.get_dna_spec_size()
        self.updated_dna = True

    def __getitem__(self, item: int) -> int:
        return self.dna[item]

    def __setitem__(self, key: int, value: int):
        self.dna[key] = value

    def __len__(self):
        return len(self.dna)

    def update(self, dna_spec: List[Tuple[int, int, int]]):
        for start, end, seg in dna_spec:
            self.dna[start:end] = seg
        self.updated_dna = True

    def unset_updated_dna(self):
        self.updated_dna = False

    def get_segment_types(self) -> torch.Tensor:
        return torch.unique(self.dna).type(torch.uint8).to(self.device)

    def get_mask(self, seg: Union[int, torch.Tensor]) -> torch.Tensor:
        if isinstance(seg, int):
            seg = torch.tensor(seg)
        if torch.any(seg == UNSPECIFIC_ID):
            seg = torch.arange(torch.max(self.dna) + 1)
        return torch.isin(self.dna, seg.to(self.device))

    def get_bins(self) -> torch.Tensor:
        bins = torch.where(self.dna[1:] != self.dna[:-1])[0] + 1
        bins = torch.cat([
            torch.zeros(1, dtype=bins.dtype),
            bins,
            torch.tensor([self.size], dtype=bins.dtype)]
        ).to(self.device)
        return bins

    def get_segment_array(self) -> torch.Tensor:
        bins = self.get_bins()
        segment_array = torch.tensor(self.size, dtype=torch.uint8).to(self.device)
        prev = bins[0]
        for num, next in enumerate(bins[1:]):
            segment_array[prev:next] = num
            prev = next
        return segment_array

    def get_dna_spec_size(
            self,
            mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(self.size, dtype=torch.bool).to(self.device)

        spec_size = torch.zeros(
            torch.max(self.get_segment_types()) + 1,
            dtype=torch.float16
        ).to(self.device)
        for seg in self.get_segment_types():
            spec_size[seg] += torch.sum(torch.logical_and(mask, self.get_mask(seg)))
        return spec_size

    def get_state_distribution(self) -> torch.Tensor:
        return self.dna_spec_size / torch.sum(self.dna_spec_size)

    def set_device(self, device):
        self.device = device
        self.dna = self.dna.to(self.device)
        self.dna_spec_size = self.dna_spec_size.to(self.device)


