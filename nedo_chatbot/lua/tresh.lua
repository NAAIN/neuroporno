function ReloadDB()
 db:close()
 db = sqllib.open(Dbfilename)
 print("ее бд релоадед")
 stmt = db:prepare[[ select * from tresh where inp = :question_name ]]
 return true
end

function DropDB()
 db:exec('DROP TABLE tresh')
 db:exec('CREATE TABLE tresh (inp TEXT,answ TEXT,is_shell INTEGER,is_int_comm INTEGER)')
 print("бдшке пиздец <3")
 ReloadDB(Dbfilename)
 return true
end

function haragiri()
 FreeDB()
 os.exit()
 return true
end

function FreeDB()
 db:close()
 return true
end

function LoadDB()
 db = sqllib.open(Dbfilename)
 print("ее бд лоадед")
 stmt = db:prepare[[ select * from tresh where inp = :question_name ]]
 return true
end
