from keras import models
from keras import layers
from keras import optimizers


#
#
#  Idea 1,
#  Parse the CSV file with all the information add the pictures to a array or data structure and fetch them from there?
#
#  Idea 2.
#  Make lots of folders and do it that way?
#
#  Idea 3.
#  TBD.
#
class datafetcher():
        def __init__(self, folder):
            self.folder = folder

        def get_family(self, family):
            return 0;


        def get_subfamily(self, family):
            return 0;


        def get_speices(self, family):
            return 0;


        def get_genus(self, family):
            return 0;

class MyModel(keras.model):

    def __init__(self):
        super(MyModel, self).__init___()
        self.dense1 = layers.Dense(20, actication='relu')
        self.dense2 = layers.Dense(20, actication='relu')
        self.dense3 = layers.Dense(10, actication='softmax')

    def call(self,inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x



model = MyModel
model.fit()
