import torch
import torch.nn as nn
from torch.distributions import Beta, Dirichlet
import math

from policy import SimplePolicy


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_branches, n_firms, limit=False):
        super().__init__()
        self.size = n_branches, n_firms
        print(input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.buy = nn.Sequential(
            nn.Linear(hidden_dim, n_firms * n_branches),
            nn.Softmax(dim=-1),
            nn.Unflatten(-1, (n_firms, n_branches)),
        )
        self.sale = nn.Sequential(nn.Linear(hidden_dim, n_branches), nn.Sigmoid())
        self.use = nn.Sequential(
            nn.Linear(hidden_dim, n_branches + 1 if not limit else 2 * n_branches + 1),
            nn.Sigmoid(),
        )
        self.prices = nn.Sequential(nn.Linear(hidden_dim, n_branches), nn.Sigmoid())

    def forward(
            self, y
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x:
        :return: tuple[
            percent_to_buy: [n_firms, n_branches],
            percent_to_sale: [n_branches],
            percent_to_use: [n_branches],
            percent_price_change: [n_branches]
        ]
        """
        x = self.net(y)
        percent_to_buy = self.buy(x)
        percent_to_sale = self.sale(x)
        percent_to_use = self.use(x)[:-1]
        percent_price_change = self.prices(x)
        return percent_to_buy, percent_to_sale, percent_to_use, percent_price_change


class BetaPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_branches, n_firms, limit=False):
        super().__init__()
        self.size = n_firms, n_branches

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.buy = nn.Sequential(
            nn.Linear(hidden_dim, n_firms * n_branches + 1),
            nn.Softplus(),
        )
        self.sale = nn.Sequential(
            nn.Linear(hidden_dim, 2 * n_branches),
            nn.Softplus(),
            nn.Unflatten(-1, (n_branches, 2))
        )
        self.use = nn.Sequential(
            nn.Linear(hidden_dim, 2 * (n_branches + 1 if not limit else 2 * n_branches + 1)),
            nn.Softplus(),
            nn.Unflatten(-1, (n_branches + 1 if not limit else 2 * n_branches + 1, 2))
        )
        self.prices = nn.Sequential(
            nn.Linear(hidden_dim, 2 * n_branches),
            nn.Softplus(),
            nn.Unflatten(-1, (n_branches, 2))
        )

    def forward(self, y, eps=1e-9):
        x = self.net(y)
        buy_params = self.buy(x).clamp(min=eps)
        sale_params = self.sale(x).clamp(min=eps)
        use_params = self.use(x).clamp(min=eps)
        prices_params = self.prices(x).clamp(min=eps)

        percent_to_buy = Dirichlet(buy_params).rsample()[:-1].unflatten(-1, self.size)
        percent_to_sale = Beta(sale_params[..., 0], sale_params[..., 1]).rsample()
        percent_to_use = Beta(use_params[..., 0], use_params[..., 1]).rsample()[:-1]
        percent_price_change = Beta(prices_params[..., 0], prices_params[..., 1]).rsample()

        return percent_to_buy, percent_to_sale, percent_to_use, percent_price_change


class PolicyNet(SimplePolicy):
    def __init__(
            self,
            firm,
            net=PolicyNetwork,
            hidden_dim=128,
    ):
        super().__init__(firm)
        input_dim = self.get_input_dim(firm.market)
        print(f"Input_dim: {input_dim}")
        n_branches = firm.n_branches
        n_firms = firm.market.n_firms
        self.net = net(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_branches=n_branches,
            n_firms=n_firms,
        )

    def get_input_dim(self, market) -> int:
        return 2 * math.prod(market.price_matrix.shape) + sum(market.price_matrix.shape) + 1

    def __call__(
            self, market
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param market:
        :return: tuple[
            percent_to_buy: [n_firms, n_branches],
            percent_to_sale: [n_branches],
            percent_to_use: [n_branches],
            prices: [n_branches]
        ]
        """
        state = torch.concatenate(
            [
                market.price_matrix.flatten(),
                market.volume_matrix.flatten(),
                market.gains,
                self.firm.reserves.flatten(),
                self.firm.financial_resources.unsqueeze(0),
            ]
        ).type(torch.float32)
        (
            percent_to_buy,
            percent_to_sale,
            percent_to_use,
            percent_price_change,
        ) = self.net(state)
        prices = (market.price_matrix[self.firm.id] * (1 + percent_price_change)).type(
            market.dtype
        )
        return percent_to_buy, percent_to_sale, percent_to_use, prices
