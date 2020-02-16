import numpy as np



#auxiliary classes for managing features
class first_feature():
    def __init__(self, index_briscola):
        # first feature, matrix keeping track of cards used, every row corresponds to a seed
        self.deck_state = np.ones((4, 10)).astype(int)
        self.index_briscola = index_briscola
        self.num_cards=40



    def register(self, card):
        i = int(card / 10)
        j = card % 10
        self.deck_state[i, j] = 0
        self.num_cards = self.num_cards - 1


    def create(self, card):
        i = int(card / 10)
        j = card % 10
        num_cards_stronger = sum(self.deck_state[i, j + 1:10])

        if i != self.index_briscola:
            num_cards_stronger = num_cards_stronger + sum(self.deck_state[self.index_briscola, 0:10])

        return num_cards_stronger / self.num_cards


class second_feature():
    def __init__(self):
        self.total_points=120
        self.points=[0, 0, 0, 0, 0, 2, 3, 4, 10, 11]

    def register(self, card):
        j=card%10
        self.total_points-self.points[j]

    def create(self, card):
        j=card%10
        if self.total_points==0:
            y=0
        else:
            y=self.points[j]/self.total_points
        return y
