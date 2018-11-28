from model import *
from trainer import Trainer

def main():
  fe = FrontEnd()
  d = D()
  q = Q()
  g = G()

  for i in [fe, d, q, g]:
    i.cuda()
    i.apply(weights_init)

  trainer = Trainer(g, fe, d, q)
  trainer.train()

if __name__ == '__main__':
    main()


