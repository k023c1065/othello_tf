#!/usr/bin/python3.6
import sys
error=' '

import subprocess
import io
#日本語対応
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cgi
# cgiはcgiプログラムに使うモジュール。
import cgitb
# cgitbはcgiプログラムデバッグに関するモジュール。うまくいかない。
cgitb.enable()
import hashlib

hashed_password="332d9fb28aee538613db85461c76e06933b657a22a47cfdcef4989f3484a2184"
def is_password(password):
    password=hashlib.sha256(password.encode()).hexdigest()
    return password == hashed_password
name=' '
toukou=' '
# パラメータを取得するための関数
# get、post区分なしでデータを持ち込む。
form = cgi.FieldStorage()
# パラメータを取得する。
result=' '
if len(form) > 0:
  name = form.getvalue('name','')
  password = form.getvalue('password','')
  if is_password(password):
    r=subprocess.run(name,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    result="stdout:<br>"+r.stdout.decode().replace("\n","<br>")+"<br>stderr:<br>"+r.stderr.decode().replace("\n","<br>")
  else:
    result="Password did not matched<br>"
# 画面応答するhtmlドキュメント
html = f"""	
<!DOCTYPE html>	
<html>	
  <head><title>formで送る</title></head>	
  <body>
  error:{error}<br>
  <h1>formで送る</h1>
    <form action='' method='post'>
    <label for="user_name">コマンド:</label>
    <input type='text' id="user_name" name='name' value=''><br><br>	
    <label for="user_name">パスワード:</label>
    <input type='password' id="password" name='password' value=''><br><br>	
    <input type="submit" value="送信">
    </form>	


    <h2>内容</h2>
    結果-<br> {result} 

""";	
# ヘッダータイプ設定
print("Content-type: text/html; charset=UTF-8")
# httpプロトコールでheaderとbodyの区分は改行なので必ず入れる。なければエラーに発生する。(bodyがないhttpファイルなので)
print('')
# バーディーを出力
print(html)
print("</body></html>")