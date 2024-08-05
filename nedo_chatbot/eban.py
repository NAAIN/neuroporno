import subprocess
import signal
from telethon import TelegramClient, events
with open("config.txt") as ConfF:
 api_id = ConfF.readline().rstrip('\n')
 api_hash = ConfF.readline().rstrip('\n')
 Conf_bot_token = ConfF.readline().rstrip('\n')
 ConfF.close()

bot = TelegramClient('bot', int(api_id), api_hash).start(bot_token=Conf_bot_token)
process = subprocess.Popen(["lua", "./lua/main.lua"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

signal.signal(signal.SIGINT, signal.SIG_IGN)

@bot.on(events.NewMessage())
async def comm(event):
    await event.respond(SAYGEXLUA(event.text))
    print("еее боту "+event.text+" прилетело")
    raise events.StopPropagation

def SAYGEXLUA(user_input):
 try:
  process.stdin.write(user_input.encode() + b"\n")
  process.stdin.flush()
  response = process.stdout.readline().decode().strip()
  print("ее отправляем кому то "+f"{response}")
  if f"{response}" == "":
   return "луашка мразота ничо не дала(9(99("
  else:
   return f"{response}"
 except BrokenPipeError:
    bot.disconnect()
    return "блядь я себя захуярил (BrokenPipeError)"

def main():
    bot.run_until_disconnected()

def term_handler(signum, frame):
  print("ураа ctrl+c")
  process.send_signal(signal.SIGINT)
  process.wait()
  exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, term_handler)
    main()

