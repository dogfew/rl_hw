import torch
import torch.nn as nn
from typing import Callable


class Market:
    def __init__(
            self, n_branches: int = 2,
            min_price: float = 1,
            base_price: float = 1000,
            max_price: float = 1_000_000_000,
            dtype=torch.int64
    ):
        """
        :param n_branches: Число отраслей
        """
        self.n_branches: int = n_branches
        self.dtype = dtype
        self.volume_matrix: torch.Tensor = torch.zeros(
            (0, n_branches), dtype=self.dtype
        )
        self.price_matrix: torch.Tensor = torch.empty(
            (0, n_branches), dtype=self.dtype
        )
        self.gains: torch.Tensor = torch.zeros(0, dtype=self.dtype)
        self.MIN_PRICE = min_price
        self.BASE_PRICE = base_price
        self.MAX_PRICE = max_price
        self.max_id = 0

    @property
    def n_firms(self):
        return self.max_id

    @property
    def total_volumes(self):
        return self.volume_matrix.sum(dim=0)

    def __add_row(self) -> None:
        """
        Добавить строки в v_matrix, p_matrix и gains
        """
        self.volume_matrix = torch.cat(
            (
                self.volume_matrix,
                torch.zeros((1, self.n_branches), dtype=self.dtype),
            ),
            dim=0,
        )
        self.price_matrix = torch.cat(
            (
                self.price_matrix,
                torch.full((1, self.n_branches), self.BASE_PRICE, dtype=self.dtype),
            ),
            dim=0,
        )
        self.gains = torch.cat(
            (self.gains, torch.tensor([0], dtype=self.dtype)),
            dim=0,
        )

    def generate_id(self):
        self.__add_row()
        self.max_id += 1
        return self.max_id

    def process_purchases(
            self, purchase_matrix: torch.Tensor, sellers_gains: torch.Tensor
    ):
        self.volume_matrix -= purchase_matrix
        self.gains += sellers_gains

    def process_sales(self, firm_id: int, volumes: torch.Tensor):
        self.volume_matrix[firm_id] += volumes

    def process_gains(self, firm_id: int):
        firms_gain = self.gains[firm_id].item()
        self.gains[firm_id] -= firms_gain
        assert self.gains[firm_id] == 0
        return firms_gain

    def process_prices(self, firm_id: int, new_prices: torch.Tensor):
        self.price_matrix[firm_id] = torch.clip(new_prices, self.MIN_PRICE, self.MAX_PRICE)

    def __repr__(self):
        representation = (
            f"Market(n_firms: {self.n_firms}, n_commodities: {self.n_branches})"
            f"\n\tVolume Matrix:\n{self.volume_matrix}"
            f"\n\tPrice Matrix:\n{self.price_matrix}"
            f"\n\tGain Matrix:\n{self.gains}"
        )
        return representation
