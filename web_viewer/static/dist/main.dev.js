"use strict";

var stage = document.getElementById("stage");
var squareTemplate = document.getElementById("square-template");
var stoneStateList = [];
var currentColor = 1;
var currentTurnText = document.getElementById("current-turn");
var passButton = document.getElementById("pass");

function changeTurn() {
  currentColor = 3 - currentColor;

  if (currentColor === 1) {
    currentTurnText.textContent = "黒";
  } else {
    currentTurnText.textContent = "白";
  }

  console.log("Color::", currentColor);
}

function skipBlackTurn() {
  currentColor = 3 - currentColor;

  if (currentColor === 1) {
    currentTurnText.textContent = "黒";
  } else {
    currentTurnText.textContent = "白";
  }

  console.log("Color::", currentColor);
}

function getReversibleStones(idx) {
  //クリックしたマスから見て、各方向にマスがいくつあるかをあらかじめ計算する
  //squareNumsの定義はやや複雑なので、理解せずコピーアンドペーストでも構いません
  var squareNums = [7 - idx % 8, Math.min(7 - idx % 8, (56 + idx % 8 - idx) / 8), (56 + idx % 8 - idx) / 8, Math.min(idx % 8, (56 + idx % 8 - idx) / 8), idx % 8, Math.min(idx % 8, (idx - idx % 8) / 8), (idx - idx % 8) / 8, Math.min(7 - idx % 8, (idx - idx % 8) / 8)]; //for文ループの規則を定めるためのパラメータ定義

  var parameters = [1, 9, 8, 7, -1, -9, -8, -7]; //ここから下のロジックはやや入念に読み込みましょう
  //ひっくり返せることが確定した石の情報を入れる配列

  var results = []; //8方向への走査のためのfor文

  for (var i = 0; i < 8; i++) {
    //ひっくり返せる可能性のある石の情報を入れる配列
    var box = []; //現在調べている方向にいくつマスがあるか

    var squareNum = squareNums[i];
    var param = parameters[i]; //ひとつ隣の石の状態

    var nextStoneState = stoneStateList[idx + param]; //フロー図の[2][3]：隣に石があるか 及び 隣の石が相手の色か -> どちらでもない場合は次のループへ

    if (nextStoneState === 0 || nextStoneState === currentColor) continue; //隣の石の番号を仮ボックスに格納

    box.push(idx + param); //フロー図[4][5]のループを実装

    for (var j = 0; j < squareNum - 1; j++) {
      var targetIdx = idx + param * 2 + param * j;
      var targetColor = stoneStateList[targetIdx]; //フロー図の[4]：さらに隣に石があるか -> なければ次のループへ

      if (targetColor === 0) continue; //フロー図の[5]：さらに隣にある石が相手の色か

      if (targetColor === currentColor) {
        //自分の色なら仮ボックスの石がひっくり返せることが確定
        results = results.concat(box);
        break;
      } else {
        //相手の色なら仮ボックスにその石の番号を格納
        box.push(targetIdx);
      }
    }
  } //ひっくり返せると確定した石の番号を戻り値にする


  console.log("get_rev_result:", idx, results);
  return [idx, results];
}

;

function EncodeHTMLForm(data) {
  var params = [];

  for (var name in data) {
    var value = data[name];
    var param = encodeURIComponent(name) + '=' + encodeURIComponent(value);
    params.push(param);
  }

  return params.join('&').replace(/%20/g, '+');
}

;

function get_move() {
  var form, formResponse;
  return regeneratorRuntime.async(function get_move$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          // 1. フォーム（multipart/form-data）形式でPOSTする
          // 2. FormDataクラスのインスタンスを作成する
          form = new FormData();
          form.append('board', stoneStateList); // 3. Post通信

          _context.next = 4;
          return regeneratorRuntime.awrap(fetch("/model", {
            method: "POST",
            // HTTP-Methodを指定する！
            body: form // リクエストボディーにフォームデータを設定

          }));

        case 4:
          formResponse = _context.sent;
          _context.next = 7;
          return regeneratorRuntime.awrap(formResponse.text());

        case 7:
          return _context.abrupt("return", _context.sent);

        case 11:
          _context.t1 = _context.sent;

          _context.t0.log.call(_context.t0, _context.t1);

        case 13:
        case "end":
          return _context.stop();
      }
    }
  });
}

function place_stone(res) {
  index = res[0]; //ひっくり返せる石の数を取得

  var reversibleStones = res[1];
  console.log("flag:", reversibleStones, stoneStateList[index]);
  console.log("flag bool:", stoneStateList[index] !== 0, !reversibleStones.length); //他の石があるか、置いたときにひっくり返せる石がない場合は置けないメッセージを出す

  if (stoneStateList[index] !== 0 || !reversibleStones.length) {
    alert("ここには置けないよ！");
    return "";
  } //自分の石を置く 


  stoneStateList[index] = currentColor;
  document.querySelector("[data-index='".concat(index, "']")).setAttribute("data-state", currentColor); //相手の石をひっくり返す = stoneStateListおよびHTML要素の状態を現在のターンの色に変更する

  reversibleStones.forEach(function (key) {
    stoneStateList[key] = currentColor;
    document.querySelector("[data-index='".concat(key, "']")).setAttribute("data-state", currentColor);
  });
}

function onClickSquare(index) {
  return regeneratorRuntime.async(function onClickSquare$(_context2) {
    while (1) {
      switch (_context2.prev = _context2.next) {
        case 0:
          console.log("my_index:", index);
          console.log("Color:", currentColor);

          if (!(currentColor == 2)) {
            _context2.next = 4;
            break;
          }

          return _context2.abrupt("return");

        case 4:
          r = getReversibleStones(index);
          console.log("get_reversible_stones:", r);
          r = place_stone(r);

          if (!(r === "")) {
            _context2.next = 9;
            break;
          }

          return _context2.abrupt("return");

        case 9:
          //ゲーム続行なら相手のターンにする
          console.log("Color:", currentColor);
          r = '';
          get_move().then(function (index) {
            return new Promise(function (resolve, reject) {
              setTimeout(function () {
                changeTurn();
                console.log("Color;", currentColor);
                resolve(index);
              }, 20);
            });
          }).then(function (index) {
            return new Promise(function (resolve, reject) {
              setTimeout(function () {
                r = index.split(",");
                console.log("API res:", index);
                index = Number(r[0]) * 8 + Number(r[1]);
                console.log("enemy index", index);
                console.log("Color:", currentColor);
                r_list = getReversibleStones(index);
                console.log("r_list", r_list);
                resolve(r_list);
              }, 20);
            });
          }).then(function (r_list) {
            place_stone(r_list);
          }).then(function () {
            changeTurn();
          }); //ゲーム続行なら相手のターンにする

        case 12:
        case "end":
          return _context2.stop();
      }
    }
  });
}

function createSquares() {
  var _loop, i;

  return regeneratorRuntime.async(function createSquares$(_context3) {
    while (1) {
      switch (_context3.prev = _context3.next) {
        case 0:
          _loop = function _loop(i) {
            var square = squareTemplate.cloneNode(true);
            square.removeAttribute("id");
            stage.appendChild(square);
            var stone = square.querySelector('.stone');
            var defaultState = void 0; //iの値によってデフォルトの石の状態を分岐する

            if (i == 27 || i == 36) {
              defaultState = 1;
            } else if (i == 28 || i == 35) {
              defaultState = 2;
            } else {
              defaultState = 0;
            }

            stone.setAttribute("data-state", defaultState);
            stone.setAttribute("data-index", i); //インデックス番号をHTML要素に保持させる

            stoneStateList.push(defaultState); //初期値を配列に格納

            square.addEventListener('click', function () {
              onClickSquare(i);
            });
          };

          for (i = 0; i < 64; i++) {
            _loop(i);
          }

        case 2:
        case "end":
          return _context3.stop();
      }
    }
  });
}

window.onload = function () {
  createSquares();
  passButton.addEventListener("click", changeTurn);
  console.log("Color:", currentColor);
  r = '';
  get_move().then(function (index) {
    return new Promise(function (resolve, reject) {
      setTimeout(function () {
        changeTurn();
        console.log("Color;", currentColor);
        resolve(index);
      }, 20);
    });
  }).then(function (index) {
    return new Promise(function (resolve, reject) {
      setTimeout(function () {
        r = index.split(",");
        console.log("API res:", index);
        index = Number(r[0]) * 8 + Number(r[1]);
        console.log("enemy index", index);
        console.log("Color:", currentColor);
        r_list = getReversibleStones(index);
        console.log("r_list", r_list);
        resolve(r_list);
      }, 20);
    });
  }).then(function (r_list) {
    place_stone(r_list);
  }).then(function () {
    changeTurn();
  });
};