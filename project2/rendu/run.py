
import conv_net
from flip_training import flip_training
from test_set_formatting import test_set_formatting


def run():
    flip_training()
    test_set_formatting()
    conv_net.execute(restore_flag = True)






if __name__ == '__main__':
    run()
