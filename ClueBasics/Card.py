class Card:

        def __init__(self, type, name):
                self.type = type
                self.name = name
                
        def getType(self):
            return self.type
        
        def getName(self):
            return self.name
        
        def __str__(self):
            return self.name        
        
        def __repr__(self):
            return str(self)