# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:18:42 2026

@author: janas
"""
from PIL import Image, ImageDraw, ImageFont
import os

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

learner = "GRAMMAR\nPROFESSIONAL"
#"VOCABULARY\nPROFESSIONAL"
#"ALLROUND\nLEARNER"  #dynamish

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

img = Image.open("app_slides/slide3.png")
draw = ImageDraw.Draw(img)

width, height = img.size

# Fonts
font_title = ImageFont.truetype("arialbd.ttf", int(height*0.065))
font_desc = ImageFont.truetype("arial.ttf", int(height*0.035))

# ---- TITLE ----
draw.text(
    (width/2, height*0.23),   # positie midden boven duo
    learner,
    fill="#63E000",
    font=font_title,
    anchor="mm", 
    align="center"
)

# ---- DESCRIPTION ----
draw.text(
    (width/2, height*0.70),   # onder duo
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



################################################
#SLIDE 4
img = Image.open("slide4.png")
draw = ImageDraw.Draw(img)

width, height = img.size

learning_age = 27  # dynamisch

font = ImageFont.truetype("arialbd.ttf", 400)

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



