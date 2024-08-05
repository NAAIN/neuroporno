local signal = require("posix.signal")
local exec = require 'os'.execute
local logic = require("tresh")
sqllib = require('lsqlite3')
Dbfilename = "QnA.sqlite"
db = sqllib.open(Dbfilename)

function FindInput(inp)
 if inp == nil then return "а давайте без nilов" end
 if inp:sub(1,1) == "#" then return assert(load(inp:sub(2)))() end 
 
 stmt = db:prepare[[ select * from tresh where inp = :question_name ]]
 stmt:bind_names{question_name = inp}
 if stmt:step() == 101 then return "А на такие входы меня не дресировали, идите тырить разраба чтоб добавил <3" end
 _,answ,is_shell,is_int_comm = stmt:get_uvalues()
 
 if is_int_comm == 1 then assert(load(answ))() stmt:reset() return true end
 if is_shell == 0 or is_shell == nil then stmt:reset() return answ end
 if is_shell == 1 then stmt:reset() return exec(answ) end
end

signal.signal(signal.SIGINT, function(signum)
  db:close() 
  print("\nЗакрываю бдшку")
  os.exit(0)
end)

while 1 do
print(FindInput(io.read()))
end
