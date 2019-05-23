# there are the data generators

data generators classes for model training, each generator should have the api functions as follow or inherit the Generator class(proposal)

- **\_\_len\_\_(self)** the function to return data number in the generator
- **\_\_getitem\_\_(self, index)** the function to return a batch data indicated by **index** param
