import briscola_tree as bt
import pygame
import numpy as np
import random
from Neal_NN import NN
from briscola_features import first_feature, second_feature

'''
This is the user interface to test the Network, once trained, it can also be used to train the decision trees.
To play a card, click on it, than to play the hand press any key.
'''

bg=bt.tree()

pygame.init()

#This determines wether you are playing the Network or the Tree
neal_plays=True

first_NN=NN( L=3, H=8, I=2, O=1, normalized=1, alpha=1, beta=1, centered=True)
second_NN=NN( L=3, H=8, I=4, O=1, normalized=1, alpha=1, beta=1, centered=True)
#This loads the trained models. That are two networks, one for when playing first, the other for playing second
first_NN.load_parameters('briscola_bot1')
second_NN.load_parameters('briscola_bot2')



#Here set the directory where you put the images with the cards
address="your directory"
images=[]
for card in bg.actual_deck:
    name_img=address+"\\"+card.name+"_of_"+card.suit+".png"
    images.append(pygame.image.load(name_img))


SIZE_IMG=images[0].get_rect().size
rescale=0.2
width_card=SIZE_IMG[0]*rescale
height_card=SIZE_IMG[1]*rescale

for i in range(40):
    images[i] = pygame.transform.scale(images[i], (round(SIZE_IMG[0]*rescale),round(SIZE_IMG[1]*rescale)))

WIDTH=800
HEIGHT=700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
ongoing = True


white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0,0,0)
backdrop_color = [0, 100, 0]
font = pygame.font.Font('freesansbold.ttf', 32)
title = font.render('Game of Briscola', True, green, blue)
briscola_text=font.render('Briscola',True,green,blue)

x_briscola_text=width_card/2
y_briscola_text=HEIGHT/2-height_card/2

title_section=HEIGHT*0.1
x_title=WIDTH/2
y_title=title_section/2

titleRect = title.get_rect()
titleRect.center = (x_title,y_title)

card_section=height_card*1.1
x_card_left=WIDTH/4
x_card_middle=WIDTH/2
x_card_right=WIDTH*3/4
y_cards_player2=title_section
y_card_played=[]
y_card_played.append(title_section+2*card_section)
y_card_played.append(title_section+card_section)
y_cards_player1=title_section+3*card_section

x_cards=[x_card_left,x_card_middle,x_card_right]

scores = [0, 0]
# shuffle deck
deck = list(range(40))
# pick the briscola
last_card = deck.pop(random.randint(0, 39))
bg.briscola = bg.actual_deck[last_card].suit
random.shuffle(deck)
deck.append(last_card)

#initialize creators of features
suits = ["hearts", "diamonds", "clubs", "spades"]
index_briscola=suits.index(bg.briscola)
x_=first_feature(index_briscola)
y_ = second_feature()




bg.first_hand = True
# determine who is first to go
win = random.choice([0, 1])

hand1 = []
hand2 = []

phases=["draw","play_first","play_second","wait"]
it_phase=0

first_card = []
second_card = []

progress=False


screen.fill(backdrop_color)

while ongoing:

    screen.blit(title, titleRect)

    index_choice = -1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            ongoing=False
        elif event.type==pygame.KEYDOWN and it_phase==3:
            progress=True
        elif event.type == pygame.MOUSEBUTTONDOWN:

            mouse=event.pos

            if event.button == 1:
                if mouse[1] <= y_cards_player1 + height_card and mouse[
                    1] >= y_cards_player1:
                    if mouse[0] >= x_card_left and mouse[0] <= x_card_left + width_card:
                        index_choice=0
                    if mouse[0] <= x_card_middle + width_card  and mouse[0] >= x_card_middle:
                        index_choice=1
                    if mouse[0] <= x_card_right + width_card and mouse[0] >= x_card_right:
                        index_choice=2

    if np.shape(deck)[0]>0:
        screen.blit(briscola_text,(x_briscola_text,y_briscola_text-50))
        screen.blit(images[last_card],(x_briscola_text,y_briscola_text))


    if phases[it_phase]=="draw":
        if deck:
            if bg.first_hand:
                bg.first_hand = False
                for k in range(3):
                    hand1.append(deck.pop(0))
                    hand2.append(deck.pop(0))
                    x_.register(hand2[-1])

            else:
                hand1.append(deck.pop(0))
                hand2.append(deck.pop(0))
                x_.register(hand2[-1])



        for i in range(np.shape(hand1)[0]):
            screen.blit(images[hand1[i]], (x_cards[i], y_cards_player1))

        for i in range(np.shape(hand2)[0]):
            screen.blit(images[hand2[i]], (x_cards[i], y_cards_player2))


        first_card = []
        second_card = []
        it_phase = (it_phase+1)%4



    if phases[it_phase]=="play_first":
        if win == 0:
            # in this simulation player1 is the human
            if not index_choice==-1 and not index_choice>np.shape(hand1)[0]-1:
                first_card=hand1[index_choice]
                hand1.remove(first_card)

        if win == 1:
            if neal_plays:
                succ_perc = []
                for card in hand2:
                    x = x_.create(card)
                    y = y_.create(card)
                    y_pred = first_NN.predict(x_pred=np.expand_dims([x, y], 1))
                    succ_perc.append(y_pred[0])
                first_card = hand2[np.argmax(succ_perc)]

            else:
                first_card = bg.decide_first_move(deck[:], hand1[:], hand2[:], win, bg.simulations, bg.depth)

            hand2.remove(first_card)

        if not first_card==[]:
            screen.blit(images[first_card], (x_card_middle, y_card_played[win]))
            it_phase = (it_phase + 1) % 4





    if phases[it_phase]=="play_second":
        if win == 1:
            # in this simulation player1 is the human
            if not index_choice==-1 and not index_choice>np.shape(hand1)[0]-1:
                second_card = hand1[index_choice]
                hand1.remove(second_card)

        if win == 0:
            if neal_plays:
                succ_perc = []
                x_other_ = x_.create(first_card)
                y_other_ = y_.create(first_card)
                for card in hand2:
                    _, new_win = bg.play_hand(first_card, card, win, scores[:])
                    hand_won = 2 * np.float64(new_win != win) - 1
                    x_other = x_other_ * hand_won
                    y_other = y_other_ * hand_won
                    x_reply = x_.create(card) * hand_won
                    y_reply = y_.create(card) * hand_won

                    y_pred = second_NN.predict(
                        x_pred=np.expand_dims([x_reply, y_reply, x_other, y_other], 1))
                    succ_perc.append(y_pred[0])

                second_card = hand2[np.argmax(succ_perc)]

            else:
                second_card = bg.decide_response(first_card, deck[:], hand1[:], hand2[:], win, bg.simulations, bg.depth)

            hand2.remove(second_card)

        if not second_card==[]:
            screen.blit(images[second_card], (x_card_middle, y_card_played[(win+1)%2]))
            scores, win = bg.play_hand(first_card, second_card, win, scores)
            it_phase = (it_phase + 1) % 4

            y_.register(first_card)
            y_.register(second_card)


    if progress:
        if not hand1:
            winner=np.argmax(scores)
            winning_score=scores[winner]
            losing_score = scores[(winner+1)%2]
            screen.fill(backdrop_color)
            results = font.render('Player '+str(winner+1)+' wins '+str(winning_score)+' to '+str(losing_score), True, black)
            screen.blit(results, (WIDTH/4, HEIGHT/4))
        else:
            it_phase = (it_phase + 1) % 4
            progress=False
            screen.fill(backdrop_color)

    pygame.display.update()




print(scores)
