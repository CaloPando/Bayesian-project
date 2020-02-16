from Neal_NN import NN
from briscola_tree import tree
import random
import numpy as np
from briscola_features import first_feature, second_feature

'''
This is to train Neal's network on the matches played by the trees, every training_interval matches the network samples, than
it plays a test match and records the result on a text file, and so on. Using long training intervals works better as we still
haven't found how to effectively update the weights in different sampling sessions
'''
directory_file="insert the directory of the file where you want to record the scores"

#Cards follow this encoding: x=(number cards that can beat the card)/(card still in deck)
#and y=(points the card is worth)/(points still available)
#fitted to (points won)/120

#Two Neural Networks, one for when playing first, and one for when playing second
first_NN=NN( L=3, H=8, I=2, O=1, normalized=1, alpha=1, beta=1, centered=True)
second_NN=NN( L=3, H=8, I=4, O=1, normalized=1, alpha=1, beta=1, centered=True)




train_games=1000
suits = ["hearts", "diamonds", "clubs", "spades"]

bg=tree(simulations=20,depth=0)
bg2=tree(simulations=5,depth=0)

testing=0

training_interval=20
testing_interval=training_interval+1

x1=[]
x2 = []
y1 = []
y2 = []

x1_reply = []
x2_reply = []
y1_reply = []
y2_reply = []

x1_other = []
x2_other = []
y1_other = []
y2_other = []



for game in range(train_games):

    scores = [0, 0]
    # shuffle deck
    deck = list(range(40))
    # pick the briscola
    last_card = deck.pop(random.randint(0, 39))
    bg.briscola = bg.actual_deck[last_card].suit
    bg2.briscola = bg2.actual_deck[last_card].suit
    random.shuffle(deck)
    deck.append(last_card)

    bg.first_hand = True
    # determine who is first to go
    win = random.choice([0, 1])

    hand1 = []
    hand2 = []

    #initialize creators of features for both players
    index_briscola=suits.index(bg.briscola)
    x1_=first_feature(index_briscola)
    x2_ = first_feature(index_briscola)
    y1_ = second_feature()
    y2_ = second_feature()

    #Every tot training games I do a test one, where the Network plays against the second tree
    if game%testing_interval==0 and game!=0:
        testing=1

    for num_turns in range(20):

        if deck:
            if bg.first_hand:
                bg.first_hand = False
                for k in range(3):
                    hand1.append(deck.pop(0))
                    hand2.append(deck.pop(0))
                    x1_.register(hand1[-1])
                    x2_.register(hand2[-1])
            else:
                hand1.append(deck.pop(0))
                hand2.append(deck.pop(0))
                x1_.register(hand1[-1])
                x2_.register(hand2[-1])


        if win==0:
            if testing:
                succ_perc=[]
                for card in hand1:
                    x1=x1_.create(card)
                    y1=y1_.create(card)
                    y_pred=first_NN.predict(x_pred=np.expand_dims([x1, y1],1))
                    succ_perc.append(y_pred[0])
                first_card = hand1[np.argmax(succ_perc)]
            else:
                first_card = bg.decide_first_move(deck[:], hand1[:], hand2[:], win, bg.simulations, bg.depth)
                x1.append(x1_.create(first_card))
                y1.append(y1_.create(first_card))

            hand1.remove(first_card)
        else:
            first_card = bg2.decide_first_move(deck[:], hand1[:], hand2[:], win, bg2.simulations, bg2.depth)

            x2.append(x2_.create(first_card))
            y2.append(y2_.create(first_card))

            hand2.remove(first_card)


        if win==0:
            second_card = bg2.decide_response(first_card, deck[:], hand1[:], hand2[:], win, bg2.simulations, bg2.depth)

            if num_turns!=19:
                x2_.register(first_card)
                _, new_win = bg.play_hand(first_card, second_card, win, scores[:])
                hand_won = 2 * np.float64(new_win!=win) - 1
                x2_reply.append(x2_.create(second_card)*hand_won)
                y2_reply.append(y2_.create(second_card)*hand_won)
                x2_other.append(x2_.create(first_card)*hand_won)
                y2_other.append(y2_.create(first_card)*hand_won)
                x1_.register(second_card)
            hand2.remove(second_card)
        else:

            x1_.register(first_card)

            if num_turns == 19:
                second_card=hand1[0]
            else:
                if testing:
                    succ_perc = []
                    x1_other_ = x1_.create(first_card)
                    y1_other_ = y1_.create(first_card)
                    for card in hand1:
                        _, new_win = bg.play_hand(first_card, card, win, scores[:])
                        hand_won = 2 * np.float64(new_win != win) - 1
                        x1_other=x1_other_*hand_won
                        y1_other=y1_other_*hand_won
                        x1_reply = x1_.create(card)*hand_won
                        y1_reply = y1_.create(card)*hand_won
                        y_pred=second_NN.predict(
                            x_pred=np.expand_dims([x1_reply, y1_reply, x1_other, y1_other],1))
                        succ_perc.append(y_pred[0])
                    second_card = hand1[np.argmax(succ_perc)]
                else:
                    second_card = bg.decide_response(first_card, deck[:], hand1[:], hand2[:], win, bg.simulations,
                                                     bg.depth)

                    _, new_win = bg.play_hand(first_card, second_card, win, scores[:])
                    hand_won = 2 * np.float64(new_win!=win) - 1
                    x1_reply.append(x1_.create(second_card)*hand_won)
                    y1_reply.append(y1_.create(second_card)*hand_won)
                    x1_other.append(x1_.create(first_card)*hand_won)
                    y1_other.append(y1_.create(first_card)*hand_won)


            x2_.register(second_card)
            hand1.remove(second_card)

        y1_.register(first_card)
        y1_.register(second_card)
        y2_.register(first_card)
        y2_.register(second_card)
        

        scores, win = bg.play_hand(first_card, second_card, win, scores)

    #Training phase
    if testing:
        testing=0
        file = open(directory_file, "a")
        file.write(str(scores)+"\n")
        file.close()

        x1 = []
        x2 = []
        y1 = []
        y2 = []

        x1_reply = []
        x2_reply = []
        y1_reply = []
        y2_reply = []

        x1_other = []
        x2_other = []
        y1_other = []
        y2_other = []


    if game%training_interval==0 and game!=0:
        print(x1_other)

        z1 = [scores[0] / 120] * len(x1)
        z1_reply = [scores[0] / 120] * len(x1_reply)

        z2 = [scores[1] / 120] * len(x2)
        z2_reply = [scores[1] / 120] * len(x2_reply)

        first_NN.train(x=np.array([x1 + x2, y1 + y2]), y=np.expand_dims(z1 + z2, 0), iter=80, warmup=50, thin=2,
                       chains=1, cores=1)
        second_NN.train(
            x=np.array([x1_reply + x2_reply, y1_reply + y2_reply, x1_other + x2_other, 
                        y1_other + y2_other])
            , y=np.expand_dims(z1_reply + z2_reply, 0), iter=80, warmup=50, thin=2, chains=1, cores=1)


        first_NN.save_parameters('briscola_bot1')
        second_NN.save_parameters('briscola_bot2')

