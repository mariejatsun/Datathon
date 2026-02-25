# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:18:42 2026

@author: janas
"""
from PIL import Image, ImageDraw, ImageFont
import os, textwrap
import pandas as pd

slides = []

for i in range(1, 6):
    path = os.path.join("app_slides", f"slide{i}.png")
    img = Image.open(path)
    slides.append(img)

print("Slides loaded!")

#user_id ="u:ikP3"
user_id ="u:NPs"

####################################
#slide 1

img = Image.open("app_slides/slide1.png")
img.save("app_slides_update/slide1.png")


##################################################
# SLIDE 2

df_betas_eval = pd.read_csv("betas_df_eval.csv",sep=";")

def _wrap(s, width=23):
    return "\n".join(textwrap.wrap(str(s), width=width))

def b1_message(p):
    if p >= 90: return "Memory Master.\nYou forget slower than almost everyone."
    if p >= 75: return "Strong retention.\nYour memory really sticks."
    if p >= 50: return "Solid retention.\nConsistency will boost it even more."
    if p >= 25: return "You forget a bit faster.\nShorter gaps will help."
    return "Let’s boost retention: quick reviews + steady rhythm."

def b2_message(p):
    if p >= 90: return "Speed Learner.\nRepetition works amazingly well for you."
    if p >= 75: return "Fast learner.\nYou improve quickly with practice."
    if p >= 50: return "Nice pace.\nKeep the streak going."
    if p >= 25: return "Slower gains.\nMore repetition will pay off."
    return "Let’s accelerate: smaller chunks + more frequent practice."

def draw_profile_slide(slide_in, slide_out, df_betas_eval, user_id):
    # ---- lookup ----
    row = df_betas_eval.loc[df_betas_eval["user_id"] == user_id]
    if row.empty:
        raise ValueError(f"user_id '{user_id}' not found in df_betas_eval")
    row = row.iloc[0]
    p1 = float(row["pct_b1"])
    p2 = float(row["pct_b2"])

    # ---- image ----
    img = Image.open(slide_in).convert("RGBA")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # ---- card areas (RELATIVE coords: tweak once if needed) ----
    # These assume two big green rounded rectangles centered mid-slide like your screenshot.
    left_card  = (0.12*W, 0.28*H, 0.46*W, 0.78*H)  # (x1,y1,x2,y2)
    right_card = (0.54*W, 0.28*H, 0.88*W, 0.78*H)

    # ---- fonts (scale with image height) ----
    font_hdr   = ImageFont.truetype("arialbd.ttf", int(H*0.030))
    font_pct   = ImageFont.truetype("arialbd.ttf", int(H*0.070))
    font_body  = ImageFont.truetype("arialbd.ttf",   int(H*0.024))
    font_small = ImageFont.truetype("arialbd.ttf",   int(H*0.017))

    
    draw.text((0.29*W, 0.27*H), "FORGETTING\nRATE",
          font=font_hdr, fill="white", anchor="mm", align="center")

    draw.text((0.71*W, 0.27*H), "LEARNING\nRATE",
          font=font_hdr, fill="white", anchor="mm", align="center")

    def draw_card(card, pct_value, body_text, align_side="left"):
        x1,y1,x2,y2 = card
        card_w = x2-x1
        card_h = y2-y1
    
        # horizontale uitlijning (meer naar rand)
        if align_side == "left":
            cx = x1 + 0.40*card_w
        else:
            cx = x1 + 0.60*card_w
    

        top = y1 + 0.18*card_h
    
        # percentage
        draw.text((cx, top),
                  f"{pct_value:.0f}%",
                  font=font_pct,
                  fill="white",
                  anchor="mm")
    
        # subline
        draw.text((cx, top + 0.16*card_h),
                  "Better than other learners",
                  font=font_small,
                  fill="white",
                  anchor="mm")
    
        # body – SMALLERE WRAP zodat tekst binnen kaart blijft
        wrapped = _wrap(body_text, width=18)
    
        draw.multiline_text((cx, top + 0.38*card_h),
                            wrapped,
                            font=font_body,
                            fill="white",
                            anchor="mm",
                            align="center",
                            spacing=int(H*0.008))
    
    
    draw_card(left_card,  p1, b1_message(p1), "left")
    draw_card(right_card, p2, b2_message(p2), "right")
    
    
    
    os.makedirs(os.path.dirname(slide_out) or ".", exist_ok=True)
    img.save(slide_out)
    img.save("app_slides_update/slide2.png")
    return slide_out

# ---- usage ----

draw_profile_slide("app_slides/slide2.png",
                   "app_slides_update/slide2.png",
                   df_betas_eval, user_id)

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

    return slide_out

# voorbeeld: slide3
render_profile_slide(
    slide_in="app_slides/slide3.png",
    slide_out="app_slides_update/slide3.png",
    learner=learner
)



################################################
#SLIDE 4
# 1) CSV laden (pas pad aan naar jouw bestand)
ages = pd.read_csv("user_learning_age.csv")  # of "user_learning_age_for_app.csv"


# 3) learning_age ophalen
learning_age = int(ages.loc[ages["user_id"] == user_id, "learning_age"].iloc[0])

# --- SLIDE 4 ---
img = Image.open("app_slides/slide4.png")
draw = ImageDraw.Draw(img)
width, height = img.size

font = ImageFont.truetype("arialbd.ttf", 200)


x = width / 2
y = height * 0.70

draw.text((x, y), str(learning_age),
          fill="#63E000", font=font, anchor="mm")

img.save("app_slides_update/slide4.png")




#########################################
#slide5
# cluster info laden
cluster_df = pd.read_csv("user_cluster_profiles.csv")

def get_learner_from_cluster(user_id):
    row = cluster_df.loc[cluster_df["user_id"] == user_id]
    if row.empty:
        return "UNKNOWN"

    profile = row.iloc[0]["profile"]

    mapping = {
        "REPETITION_BUILDER": "REPETITION\nBUILDER",
        "FAST_BURNER": "FAST\nBURNER",
        "STEADY_RETAINER": "STEADY\nRETAINER"
    }
    return mapping.get(profile, "UNKNOWN")

learner = get_learner_from_cluster(user_id)

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
output_path = os.path.join(output_folder, "slide5.png")
img.save(output_path)



###########################################"
#gif
paths = [f"app_slides_update/slide{i}.png" for i in range(1,6)]
imgs = [Image.open(p).convert("RGBA") for p in paths]

gif_path = "app_slides_update/slides.gif"

imgs[0].save(
    gif_path,
    save_all=True,
    append_images=imgs[1:],
    duration=1500,
    loop=0
)
# tonen
gif_path = r"C:\Users\janas\OneDrive\Documenten\KU LEUVEN\DATATHON\Datathon\app_slides_update\slides.gif"
os.startfile(gif_path) 

