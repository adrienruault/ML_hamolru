
import conv_net
from flip_training import flip_training
from test_set_formatting import convert_test_set_to_good_format


def run():
    flip_training()
    convert_test_set_to_good_format()
    
    conv_net.execute(restore_flag = True)






if __name__ == '__main__':
    run()
