# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:18:42 2026

@author: janas
"""
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd

slides = []

for i in range(1, 6):
    path = os.path.join("app_slides", f"slide{i}.png")
    img = Image.open(path)
    slides.append(img)

print("Slides loaded!")

##################################################
# SLIDE 2




#############################################
# SLIDE 3

# 1. profiles laden (1x doen)
profiles = pd.read_csv("user_profiles.csv").set_index("user_id")

# 2. functie om learner te bepalen
def get_learner_from_user(user_id):
    dominant = profiles.loc[user_id, "dominant_vg"]  # Vocabulary / Grammar / Balanced

    mapping = {
        "Balanced": "ALLROUND\nLEARNER",
        "Vocabulary": "VOCABULARY\nPROFESSIONAL",
        "Grammar": "GRAMMAR\nPROFESSIONAL"
    }
    return mapping.get(dominant, "UNKNOWN\nPROFILE")

user_id = "u:-4V"   # geselecteerde user uit dropdown
user_id = "u:-AA"
learner = get_learner_from_user(user_id)

def render_profile_slide(slide_in, slide_out, learner):
    description1= "YOU ARE BOTH EVENLY\nINVESTING IN\nVOCABULARY AND\nGRAMMAR"
    description2= "YOU ARE INVESTING\nIN VOCUBULARY, GREAT!\nBUT DON'T FORGET\nYOUR GRAMMAR"
    description3= "YOU ARE INVESTING\nIN GRAMMAR, GREAT!\nBUT DON'T FORGET\nYOUR VOCABULARY"

    if learner == "ALLROUND\nLEARNER":
        description = description1
    elif learner == "VOCABULARY\nPROFESSIONAL":
        description = description2
    elif learner == "GRAMMAR\nPROFESSIONAL":
        description = description3
    else:
        description = "UNKNOWN PROFILE"

    img = Image.open(slide_in)
    draw = ImageDraw.Draw(img)

    width, height = img.size

    # Fonts
    font_title = ImageFont.truetype("arialbd.ttf", int(height*0.065))
    font_desc  = ImageFont.truetype("arial.ttf",  int(height*0.035))

    # ---- TITLE ----
    draw.text(
        (width/2, height*0.23),
        learner,
        fill="#63E000",
        font=font_title,
        anchor="mm",
        align="center"
    )

    # ---- DESCRIPTION ----
    draw.text(
        (width/2, height*0.70),
        description,
        fill="white",
        font=font_desc,
        anchor="mm",
        align="center"
    )

    os.makedirs(os.path.dirname(slide_out), exist_ok=True)
    img.save(slide_out)
    img.show()
    return slide_out

# voorbeeld: slide3
render_profile_slide(
    slide_in="app_slides/slide3.png",
    slide_out="app_slides_update/slide3.png",
    learner=learner
)



################################################
#SLIDE 4
img = Image.open("app_slides/slide4.png")
draw = ImageDraw.Draw(img)

width, height = img.size

learning_age = 27  # dynamisch

font = ImageFont.truetype("arialbd.ttf", 200)

# positie ongeveer waar jouw 67 stond
x = width / 2
y = height * 0.70

draw.text((x, y),
          str(learning_age),
          fill="#63E000",   # duolingo groen
          font=font,
          anchor="mm")      # center-middle

img.save("learning_age_dynamic.png")
img.show()



#########################################
#slide5

learner = "REPETITION\nBUILDER"
#"FAST\nBURNER"
#"STEADY\nRETAINER"  #dynamish

description1= "YOU IGNITE FAST.\nYOU GRASP CONCEPTS\nQUICKER THAN MOST"
description2= "YOU START QUIET.\nYOU GROW STRONGER\nEVERY PRACTICE"
description3= "YOU KEEP GOING\nYOU REMEMBER\nWHAT'S IMPORTANT"

if learner == "REPETITION\nBUILDER":
    description = description2
elif learner == "FAST\nBURNER":
    description = description1
elif learner == "STEADY\nRETAINER":
    description = description3
else:
    description = "UNKNOWN PROFILE"

img = Image.open("app_slides/slide5.png")
draw = ImageDraw.Draw(img)

width, height = img.size

# Fonts
font_title = ImageFont.truetype("arialbd.ttf", int(height*0.060))
font_desc = ImageFont.truetype("arialbd.ttf", int(height*0.025))

# ---- TITLE ----
draw.text(
    (width/2+62, height*0.35),   # positie midden boven duo
    learner,
    fill="#63E000",
    font=font_title,
    anchor="mm", 
    align="center"
)

# ---- DESCRIPTION ----
draw.text(
    (width/2+60, height*0.51),   # onder duo
    description,
    fill="white",
    font=font_desc,
    anchor="mm",
    align="center"
)

output_folder = "app_slides_update"
output_path = os.path.join(output_folder, "slide_dynamic.png")
img.save(output_path)
img.show()



