import random
import numpy as np


class card:
    def __init__(self,suit,value,points,name):
        self.suit=suit
        self.value=value
        self.name=name
        self.points=points





class tree():

    def __init__(self,simulations=10,depth=2):

        self.actual_deck=self.create_deck()

        self.match_is_on=True

        self.first_hand=True

        self.num_matches=0

        self.wins=[0,0]

        self.briscola=[]

        self.depth=depth

        self.simulations=simulations

        self.depth_decrease_rate=2

        self.simulation_decrease_rate = 3

        self.penalization=1

        self.destroy_deck=True

    def create_deck(self):
        suits = ["hearts", "diamonds", "clubs", "spades"]
        names = ["2", "4", "5", "6", "7", "jack", "queen", "king", "3", "ace"]
        points = [0, 0, 0, 0, 0, 2, 3, 4, 10, 11]

        deck = []

        for suit in suits:
            it_p = 0
            for name in names:
                point = points[it_p]
                deck.append(card(suit, it_p+1, point,name))
                it_p += 1

        return deck


    def match(self):


        while self.match_is_on:
            scores=[0,0]
            #shuffle deck
            deck=list(range(40))
            # pick the briscola
            last_card=deck.pop(random.randint(0,39))
            self.briscola=self.actual_deck[last_card].suit
            random.shuffle(deck)
            deck.append(last_card)

            self.first_hand=True
            #determine who is first to go
            win=random.choice([0,1])

            hand1=[]
            hand2=[]

            for k in range(20):
                scores, win=self.turn(hand1,hand2,win,deck,self.simulations,self.depth,scores)

            print(scores)
            if(scores[0]>60):
                self.wins[0]+=1
            if (scores[1] > 60):
                self.wins[1] += 1
            else:
                self.wins += [1,1]

            self.num_matches+=1



    def turn(self,hand1,hand2,win,deck,simulations,depth,scores):
        if deck:
            if self.first_hand:
                self.first_hand = False
                for k in range(3):
                    hand1.append(deck.pop(0))
                    hand2.append(deck.pop(0))
            else:
                hand1.append(deck.pop(0))
                hand2.append(deck.pop(0))

        first_card = self.decide_first_move(deck[:], hand1[:], hand2[:], win,simulations,depth)

        if win==0:
            hand1.remove(first_card)
        else:
            hand2.remove(first_card)

        second_card = self.decide_response(first_card, deck[:], hand1[:], hand2[:], win, simulations, depth)

        if win==1:
            hand1.remove(second_card)
        else:
            hand2.remove(second_card)

        scores, win = self.play_hand(first_card, second_card, win,scores)

        return scores, win


    def play_hand(self,first_card,second_card,win,scores):

        win=self.who_wins(first_card,second_card,win)
        scores[win]+=self.actual_deck[first_card].points
        scores[win] += self.actual_deck[second_card].points

        return scores, win

    def who_wins(self,first_card,second_card,win):

        if self.actual_deck[second_card].suit==self.actual_deck[first_card].suit:
            if np.argmax([self.actual_deck[first_card].value,self.actual_deck[second_card].value])==1:
                win=(win+1)%2
        else:
            if self.actual_deck[second_card].suit==self.briscola:
                win = (win + 1) % 2

        return win


    def decide_first_move(self,deck,hand1,hand2,win,simulations,depth):

        if simulations<=0:
            hand = hand1
            if win == 1:
                hand = hand2
            return self.default_first_play(hand, win)
        my_hand=hand1
        other_hand=hand2
        if win==1:
            my_hand = hand2
            other_hand = hand1

        last_card=[]
        if deck:
            last_card = deck.pop(-1)

        deck.extend(other_hand)

        sim_scores=[]

        for first_card in my_hand:
            first_hand=my_hand[:]
            first_hand.remove(first_card)
            sim_scores.append(0)

            for i in range(simulations):
                temp_deck=deck[:]
                random.shuffle(temp_deck)

                if  last_card or last_card==0:
                    temp_deck.append(last_card)

                #I select randomly an hypothetic hand for the opponent
                second_hand=[]
                for i in range(min(np.shape(temp_deck)[0],3)):
                    second_hand.append(temp_deck.pop(0))

                second_card=self.decide_response(first_card,temp_deck[:],first_hand[:],second_hand[:],0,
                                                 0,
                                                     depth-self.depth_decrease_rate)

                #print(self.actual_deck[second_card].name+" of "+self.actual_deck[second_card].suit)

                second_hand.remove(second_card)

                sim_score=[0,0]

                sim_score, new_win = self.play_hand(first_card, second_card, win,sim_score)

                hand2 = first_hand[:]
                hand1 = second_hand[:]
                if win == 0:
                    hand1 = first_hand[:]
                    hand2 = second_hand[:]


                for k in range(depth):
                    if temp_deck:
                        sim_score, new_win = self.turn(hand1, hand2, new_win, temp_deck,
                                                        simulations-self.simulation_decrease_rate*(k+1),
                                                        depth-self.depth_decrease_rate*(k+1),sim_score)

                sim_scores[-1]=sim_scores[-1]*self.penalization+sim_score[win]-sim_score[(win+1)%2]

        card_played=my_hand[np.argmax(sim_scores)]

        return card_played



    def decide_response(self,first_card,deck,hand1,hand2,win,simulations,depth):

        if simulations<=0:
            hand=hand1
            if win==0:
                hand=hand2
            return self.default_response(first_card, hand, win)

        my_hand=hand1
        other_hand=hand2
        if win==0:
            my_hand = hand2
            other_hand = hand1

        last_card = []
        if deck:
            last_card = deck.pop(-1)
        deck.extend(other_hand)

        sim_scores=[]

        for second_card in my_hand:
            first_hand=my_hand[:]
            first_hand.remove(second_card)
            sim_scores.append(0)
            for i in range(simulations):
                temp_deck=deck[:]
                random.shuffle(temp_deck)

                if last_card or last_card==0:
                    temp_deck.append(last_card)

                #I select randomly an hypothetic hand for the opponent
                second_hand=[]

                for i in range(min(np.shape(temp_deck)[0],2)):
                    second_hand.append(temp_deck.pop(0))

                sim_score=[0,0]

                sim_score, new_win = self.play_hand(first_card, second_card, win,sim_score)

                hand2=first_hand[:]
                hand1=second_hand[:]

                if win==1:
                    hand1=first_hand[:]
                    hand2=second_hand[:]


                for k in range(depth):
                    if temp_deck:
                        sim_score, new_win = self.turn(hand1, hand2, new_win, temp_deck,
                                                        simulations-self.simulation_decrease_rate*(k+1),
                                                        depth-self.depth_decrease_rate*(k+1),sim_score)

                sim_scores[-1] = sim_scores[-1]+sim_score[(win + 1) % 2]-sim_score[win]

        card_played = my_hand[np.argmax(sim_scores)]

        return card_played



    def default_response(self,first_card,hand,win):
        #this is the null case, the card that maximizes payoff is played
        sim_scores=[]
        for second_card in hand:
            sim_scores.append(0)
            scores,_=self.play_hand(first_card,second_card,win,[0,0,0])
            sim_scores[-1] = scores[(win + 1) % 2]-scores[win]
        return hand[np.argmax(sim_scores)]

    def default_first_play(self, hand, win):
        # this is the null case, the card with the least points is played
        points = []
        for first_card in hand:
            points.append(self.actual_deck[first_card].points)

        return hand[np.argmin(points)]





