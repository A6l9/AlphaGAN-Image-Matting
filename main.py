import torch as tch

from models import AlphaGenerator
import utils as utl


def main() -> None:
    model = AlphaGenerator()

    utl.set_seed()

    x = tch.randn(1, 3, 1024, 1024)

    with tch.no_grad():
        out = model(x)


if __name__ == "__main__":
    main()
