#*  Base Layer:
class Layer:
    def __init__(self):
        self.input = None   #   A[l]
        self.output = None  #   A[l+1]
    
    def forward(self, input):
        #   TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        #   TODO: update parameters and return input gradient
        pass