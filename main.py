import tensorflow
from keras.models import load_model
import numpy as np
import cv2
import sys
import pygame
from matplotlib import pyplot as plt
from time import sleep


def draw_right_panel():
    pygame.draw.rect(screen, (72, 208, 164), pygame.Rect(screen_h, 0, screen_w - screen_h, screen_h))
    pygame.draw.rect(screen, (29, 101, 63),
                     pygame.Rect(screen_h * 10.75 // 9, screen_h * 8 // 10, (screen_w - screen_h) // 2, screen_h // 10),
                     0, 10)
    text_label = clear_font.render('Clear', True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 47 // 64, screen_h * 33 // 40))


def draw_digits_and_probabilities(prediction):
    # Biggest probability - digit
    text_label = big_font.render(str(prediction[0][1]), True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 12 // 16, screen_h // 5))
    # Biggest probability - probability
    text_label = probability_font.render(str(round(prediction[0][0]*100, 2)) + '%', True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 12 // 16 - 20, screen_h // 5 + 100))

    # second biggest probability - digit
    text_label = small_font.render(str(prediction[1][1]), True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 22 // 32, screen_h * 3 // 5))
    # second biggest probability - probability
    text_label = small_probability_font.render(str(round(prediction[1][0]*100, 2)) + '%', True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 22 // 32 - 10, screen_h * 3 // 5 + 30))

    # Third biggest probability - digit
    text_label = small_font.render(str(prediction[2][1]), True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 27 // 32, screen_h * 3 // 5))
    # Third biggest probability - probability
    text_label = small_probability_font.render(str(round(prediction[2][0]*100, 2)) + '%', True, (0, 0, 0))
    screen.blit(text_label, (screen_w * 27 // 32 - 10, screen_h * 3 // 5 + 30))


def predict_digit():
    # Obtaining screen
    photo = []
    for i in range(screen_h):
        temp = []
        for j in range(screen_h):
            pixel_color = pygame.Surface.get_at(screen, (j, i))[:-1]
            pixel_color = sum(pixel_color)/len(pixel_color)
            temp.append(pixel_color)
        photo.append(temp)
    #Prediction
    photo = np.array(photo)
    photo *= 1.0/255.0
    photo = abs(photo-1)
    photo = cv2.resize(photo, (28, 28), interpolation = cv2.INTER_AREA)
    # plt.imshow(photo, cmap='gray')
    # plt.show()
    photo = np.reshape(photo, (1, 28, 28, 1))
    prediction = my_model.predict(photo)
    print(np.argmax(prediction[0]), max(prediction[0]))

    prediction = prediction.tolist()[0]
    for i in range(10):
        prediction[i] = [prediction[i], i]
    prediction.sort(reverse=True)
    draw_digits_and_probabilities(prediction)


screen_w = 800
screen_h = screen_w*9//16
drawing_radius = 40

background_color = (255, 255, 255)

screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption('Digits Recognition')
screen.fill(background_color)

pygame.font.init()
big_font = pygame.font.SysFont('New Times Roman', 120)
probability_font = pygame.font.SysFont('New Times Roman', 40)
small_font = pygame.font.SysFont('New Times Roman', 40)
small_probability_font = pygame.font.SysFont('New Times Roman', 20)
clear_font = small_font

draw_right_panel()

previous_mouse_pos = False





# Model
my_model = load_model('../mnist_15_epochs.h5')
while True:

    if pygame.mouse.get_pressed()[0]:
        mouse_pos = pygame.mouse.get_pos()

        # Drawing digits
        if mouse_pos[0] < screen_h:
            if previous_mouse_pos:
                pygame.draw.line(screen, (0, 0, 0), previous_mouse_pos, mouse_pos, drawing_radius)
                # pygame.draw.circle(screen, (0, 0, 0), mouse_pos, drawing_radius//2)
            else:
                pygame.draw.line(screen, (0, 0, 0), mouse_pos, mouse_pos, drawing_radius)
            previous_mouse_pos = mouse_pos

        # Clicking clear button
        elif screen_h * 10.75 // 9 < mouse_pos[0] < screen_h * 14.25 // 9 and screen_h * 8 // 10 < mouse_pos[1] < screen_h * 9 // 10:
            pygame.draw.rect(screen, background_color, pygame.Rect(0, 0, screen_h, screen_h))
            draw_right_panel()
    else:
        previous_mouse_pos = False


    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            if pygame.mouse.get_pos()[0] < screen_h:
                draw_right_panel()
                predict_digit()
        elif event.type == pygame.QUIT:
            sys.exit(0)
    pygame.display.flip()
    # sleep(.1)