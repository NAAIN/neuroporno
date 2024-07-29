import sqlite3
import signal
import telethon
from telethon.tl.types import UpdateMessageReactions

with open("config.txt") as ConfF:
 api_id = ConfF.readline().rstrip('\n')
 api_hash = ConfF.readline().rstrip('\n')
 bot_token = ConfF.readline().rstrip('\n')
 bot = telethon.TelegramClient('bot', int(api_id), api_hash).start(bot_token=bot_token)
 ConfF.close()
#bot = telethon.TelegramClient('bot', int(api_id), api_hash).start(bot_token=bot_token)
signal.signal(signal.SIGINT, signal.SIG_IGN)
con = sqlite3.connect("credits.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS credit(user_id INT PRIMARY KEY UNIQUE,credit INT);")
cur.execute("CREATE TABLE IF NOT EXISTS haram(haram_word TEXT,amount INT,comment TEXT);")
cur.execute("CREATE TABLE IF NOT EXISTS halal(halal_word TEXT,amount INT,comment TEXT);")

@bot.on(telethon.events.Raw(UpdateMessageReactions))
async def handler(event):
    print(event)

@bot.on(telethon.events.NewMessage(pattern='/'))
async def comm(event):
  print("ааа комманд ивент")
  print(event)
  if event.text == "/my_credit":
    await event.reply(CreditOf(event.sender_id))
    return
  try:
    command,username = event.text.split(" ")
  except ValueError:
    return False
  if command == "/credit_of":
    for char in username:
     if char.isalpha():
       await event.reply(CreditOf(await bot.get_peer_id(username)))
       return
    await event.reply(CreditOf(username))
    return
  raise telethon.events.StopPropagation

@bot.on(telethon.events.NewMessage())
async def check_haram_words(event):
  print("ааа нью мессаге ивеит")
  print(event)
  if event.text is None: event.text == "0"
  HARAM_status = HARAMCheck(event.sender_id,event.text)
  if HARAM_status == False:
   HALAL_status = HALALCheck(event.sender_id,event.text)
   if HALAL_status == False:
     cur.execute("SELECT * FROM credit WHERE user_id = ?",(event.sender_id,))
     result = cur.fetchone()
     if result is not None:
       _,credit = result
       try:
        cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(event.sender_id,credit + 1))
       except sqlite3.Error as e:
         await event.reply(e)
     else:
       try:
        cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(event.sender_id,1))
       except sqlite3.Error as e:
         await event.reply(e)
     con.commit()
     return
   else:
     await event.reply(HALAL_status)
  else:
    await event.reply(HARAM_status)
    return
  raise telethon.events.StopPropagation

@bot.on(telethon.events.MessageEdited())
async def check_haram_words(event):
  print("аааа мессаге едит евент")
  print(event)
  if event.text is None: event.text == "0"
  HARAM_status = HARAMCheck(event.sender_id,event.text)
  if HARAM_status == False:
   HALAL_status = HALALCheck(event.sender_id,event.text)
   if HALAL_status == False:
     cur.execute("SELECT * FROM credit WHERE user_id = ?",(event.sender_id,))
     result = cur.fetchone()
     if result is not None:
       _,credit = result
       try:
        cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(event.sender_id,credit + 1))
       except sqlite3.Error as e:
         await event.reply(e)
     else:
       try:
        cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(event.sender_id,1))
       except sqlite3.Error as e:
         await event.reply(e)
     con.commit()
     return
   else:
     await event.reply(HALAL_status)
  else:
    await event.reply(HARAM_status)
    return
  raise telethon.events.StopPropagation

def CreditOf(user_id):
  # if user_id is None:
  #   return "васета /credit_of 0000000000 или /credit_of @NAAIN_heh"
  cur.execute("SELECT * FROM credit WHERE user_id = ?",(user_id,))
  result = cur.fetchone()
  if result is not None:
    _,credit = result
    return "кредитъ юзверя @" + str(user_id) + " : " + str(credit)
  else:
    return "я не знать такого юзверя!"
  
def HARAMCheck(user_id,msg):
  cur.execute("SELECT haram_word, amount, comment FROM haram")
  haram_words = cur.fetchall()
  for haram_word, amount, comment in haram_words:
      if haram_word.lower() in msg.lower():
          try:
              cur.execute("SELECT * FROM credit WHERE user_id = ?",(user_id,))
              result = cur.fetchone()
              if result is not None:
               _,credit = result
               cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(user_id,credit - amount))
               if comment == None: comment = ""
               return "ХАРАМ!1!!! -" + str(amount) + "\n" + str(comment)
              else:
               cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(user_id,1))
               return "ты новичокъ, живи!"
          except sqlite3.Error as e:
              print("Error updating credit:", e)
              return str(e)
  return False

def HALALCheck(user_id,msg):
  cur.execute("SELECT halal_word, amount, comment FROM halal")  # Retrieve all halal words at once
  halal_words = cur.fetchall()
  for halal_word, amount, comment in halal_words:
      if halal_word.lower() in msg.lower():  # Check if any halal word is present in the message
          try:
              cur.execute("SELECT * FROM credit WHERE user_id = ?",(user_id,))
              result = cur.fetchone()
              if result is not None:
               _,credit = result
               cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(user_id,credit + amount))
               con.commit()
               if comment == None: comment = ""
               return "ХАЛЯЛЬ!1!!! +" + str(amount) + "\n" + str(comment)
              else:
               cur.execute("INSERT OR REPLACE INTO credit VALUES(?,?)",(user_id,amount))
               return "ты новичокъ! +" + str(amount)
          except sqlite3.Error as e:
              print("Error updating credit:", e)
              return str(e)
  return False

def main():
    bot.run_until_disconnected()

def term_handler(signum, frame):
  print("ураа ctrl+c")
  con.commit()
  con.close()
  bot.disconnect()
  exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, term_handler)
    main()
