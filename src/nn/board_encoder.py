import torch


def convert_board_state_to_tensor(board, player_to_move, dtype=torch.float32):
    s = torch.tensor(board)

    p1 = (s == 1).to(dtype)
    p2 = (s == -1).to(dtype)
    empty = (s == 0).to(dtype)

    if player_to_move == 1:
        p3 = torch.ones_like(p1)
        p4 = torch.zeros_like(p1)
    else:
        p3 = torch.zeros_like(p1)
        p4 = torch.ones_like(p1)

    # stack channels -> shape (5, H, W)
    stacked = torch.stack([p1, p2, empty, p3, p4], dim=0)

    # add batch dim -> (1, 5, H, W)
    out = stacked.unsqueeze(0).to(dtype)

    return out
