from pathlib import Path
import csv, random
random.seed(42)
out=Path(__file__).parent/'data'/'training_reviews.csv'
rows=[]
examples={
'Positive':[
('Electronics','The phone battery lasts all day and the camera takes sharp photos.'),('Electronics','The laptop runs smoothly and starts quickly.'),('Electronics','The headphones have clear sound and comfortable ear cushions.'),('Home Appliances','The washing machine cleans clothes well and runs quietly.'),('Home Appliances','The refrigerator keeps food fresh and uses little energy.'),('Fashion & Apparel','The shirt fabric feels soft and the stitching is excellent.'),('Fashion & Apparel','These shoes are comfortable and the sizing is accurate.'),('Beauty & Personal Care','The moisturizer feels light and leaves my skin soft.'),('Food & Grocery','The snacks taste fresh and the packaging was intact.'),('Furniture & Home','The chair is sturdy, comfortable, and easy to assemble.'),('Other Products','The product is easy to use and offers great value for money.'),
],
'Negative':[
('Electronics','The phone battery drains fast and the camera quality is poor.'),('Electronics','The laptop is slow and keeps freezing during normal use.'),('Electronics','The headphones disconnect often and the sound is disappointing.'),('Home Appliances','The washing machine is noisy and leaves stains on clothes.'),('Home Appliances','The refrigerator is too noisy and arrived damaged.'),('Fashion & Apparel','The shirt fabric is poor quality and the stitching came apart.'),('Fashion & Apparel','The shoes are uncomfortable and the sizing is wrong.'),('Beauty & Personal Care','The cream caused irritation and has a strong unpleasant smell.'),('Food & Grocery','The food tasted stale and the package was leaking.'),('Furniture & Home','The chair feels flimsy and the instructions are difficult to follow.'),('Other Products','The product broke quickly and customer support was unhelpful.'),
],
'Neutral':[
('Electronics','The phone has a 6.5 inch display and 128 GB storage.'),('Electronics','The laptop arrived with a charger and a short instruction guide.'),('Electronics','The headphones are black and include a USB charging cable.'),('Home Appliances','The washing machine has several wash modes and a timer.'),('Home Appliances','The refrigerator has two shelves and a separate freezer section.'),('Fashion & Apparel','The shirt is blue and has three buttons on the front.'),('Fashion & Apparel','The shoes arrived in a cardboard box with spare laces.'),('Beauty & Personal Care','The bottle contains 100 ml and has a pump dispenser.'),('Food & Grocery','The packet contains 12 pieces and lists the ingredients.'),('Furniture & Home','The table includes four legs and assembly hardware.'),('Other Products','The package includes the product, a manual, and a warranty card.'),
]}
starts=['Honestly, ', 'In my experience, ', 'After trying it, ', 'So far, ', 'For the price, ', 'I noticed that ', 'Overall, ', '']
ends=['', ' I used it for a week.', ' This is my first impression.', ' I bought it online.', ' The description matched the item.', ' I would check this before buying again.']
for label,items in examples.items():
 for cat,txt in items:
  for i in range(10):
   prefix=random.choice(starts); suffix=random.choice(ends)
   review=(prefix+txt[0].lower()+txt[1:] if prefix else txt)+suffix
   rows.append({'review':review,'sentiment':label,'category':cat})
# Explicit mixed examples intentionally held out from training labels: these test aspect logic, not single-label model training.
with out.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['review','sentiment','category']); w.writeheader(); w.writerows(rows)
print(f'Wrote {len(rows)} examples to {out}')
