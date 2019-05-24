# there are the data generators

data generators classes for model training, each generator should have the api functions as follow or inherit the GeneratorKeras or GeneratorTf class(proposal)

- **\_\_len\_\_(self)** the function to return data number in the generator
- **\_\_getitem\_\_(self, index)** the function to return a batch tuple (batch_data, batch_target1, batch_target2, ...) indicated by **index** param
