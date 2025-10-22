const Piece = {
    King: "k",
    Queen: "q",
    Bishop: "b",
    Knight: "n",
    Rook: "r",
    Pawn: "p",

    isWhite: p => p.toUpperCase() === p,
    isBlack: p => p.toLowerCase() === p,
    areOpponents: (p1, p2) => Piece.isWhite(p1) !== Piece.isWhite(p2)
}

function make2DArray(height, width, value=0) {
    const arr = []
    for (let y=0; y<height; y++) {
        const row = []
        for (let x=0; x<width; x++) {
            row.push(value)
        }
        arr.push(row)
    }
    return arr
}

class ChessEvent {}

class GameOverEvent extends ChessEvent {
    type = "gameOver"

    constructor(outcome) {
        super()
        this.outcome = outcome
    }
}

/**
 * @typedef {object} GameOutcome
 * @property {string} termination
 * @property {"1-0" | "0-1" | "1/2-1/2"} result
 * @property {"white" | "black"} winner
 */


class Chessboard {
    static EMPTY_STATE = "8/8/8/8/8/8/8/8 w - - 0 1"
    static DEFAULT_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    static FILES = "abcdefgh"

    constructor(node) {
        /** @type {HTMLDivElement} */
        this.node = node
        this.board = this.node.querySelector(".board")
        this.state = make2DArray(8, 8, "")
        this.pieces = make2DArray(8, 8, null)
        this.moveStart = null
        this.draggedPiece = null
        this.dragPiece = this.node.querySelector("#dragged")
        this.pov = "white"
        this.whites = new Player("white")
        this.blacks = new Player("black")
        this.currentPlayer = "white"
        this.halfMoves = 0
        this.moves = 1
        this.enPassant = null
        this.promotionPopup = new Popup(document.getElementById("promotion-popup"))
        this.promotionMove = null
        this.listeners = {}
        this.history = []
        this.init()
    }

    /**
     * @param {string} event 
     * @param {ChessListener} listener 
     */
    addListener(event, listener) {
        if (this.listeners[event] === undefined) {
            this.listeners[event] = []
        }
        this.listeners[event].push(listener)
    }

    /**
     * @param {string} event 
     */
    emit(event) {
        const listeners = this.listeners[event.type]
        listeners.map(l => l(event))
    }

    init() {
        this.board.innerHTML = ""
        for (let y=0; y<8; y++) {
            for (let x=0; x<8; x++) {
                const cell = document.createElement("div")
                cell.classList.add("tile")
                cell.dataset.x = x
                cell.dataset.y = y
                cell.addEventListener("mousedown", e => this.onPickupPiece(x, y, e))
                
                if ((x + y) % 2 == 0) cell.classList.add("light")
                this.board.appendChild(cell)
            }
        }
        window.addEventListener("mouseup", e => this.onDropPiece(e))
        window.addEventListener("mousemove", e => this.updateDragPiece(e))

        Array.from(this.promotionPopup.node.querySelectorAll("button.choice")).map(btn => {
            btn.addEventListener("click", () => this.confirmPromotion(btn))
        })
        this.promotionPopup.node.querySelector("#cancel-promotion-btn").addEventListener("click", () => this.promotionPopup.hide())

        this.loadFEN(Chessboard.EMPTY_STATE)
    }

    setPOV(color) {
        if (color !== this.pov) {
            this.pov = color
            this.flip()
        }
    }

    /**
     * 
     * @param {string} fen 
     */
    loadFEN(fen) {
        this.history = []
        let [board, player, castles, enPassant, halfMoves, moves] = fen.split(" ")
        const state = make2DArray(8, 8, "")
        let x = 0
        let y = 0
        for (let i=0; i<board.length; i++) {
            const char = board[i]
            if (char === "/") {
                x = 0
                y += 1
            } else if (!isNaN(+char)) {
                x += +char
            } else {
                state[y][x] = char
                x += 1
            }
        }
        this.state = state

        this.currentPlayer = player === "w" ? "white" : "black"
        this.whites.canSmallCastle = castles.includes("K")
        this.whites.canBigCastle = castles.includes("Q")
        this.blacks.canSmallCastle = castles.includes("k")
        this.blacks.canBigCastle = castles.includes("q")

        this.enPassant = enPassant === "-" ? null : this.fromAlgebraic(enPassant)
        this.halfMoves = +halfMoves
        this.moves = +moves

        if (this.pov === "black") {
            this.flip()
        }

        this.renderPieces()
    }

    renderPieces() {
        this.pieces = make2DArray(8, 8, null)
        for (let y=0; y<8; y++) {
            for (let x=0; x<8; x++) {
                const piece = this.state[y][x]
                const tile = this.board.childNodes[y * 8 + x]
                tile.innerHTML = ""
                if (!piece) continue
                const kind = piece.toLowerCase()
                const isBlack = Piece.isBlack(piece)
                const elem = document.createElement("img")
                elem.src = `/images/${kind}.png`
                elem.classList.add("piece")
                elem.setAttribute("draggable", "true")
                if (isBlack) elem.classList.add("black")
                tile.appendChild(elem)
                this.pieces[y][x] = elem
            }
        }
    }

    onPickupPiece(x, y, event) {
        /** @type {HTMLImageElement} */
        const piece = this.pieces[y][x]
        if (!piece) return
        this.moveStart = [x, y]
        this.draggedPiece = piece
        this.updateDragPiece(event)
        this.dragPiece.style.setProperty("--size", piece.clientWidth + "px")
        this.dragPiece.src = piece.src
        this.dragPiece.classList.toggle("black", piece.classList.contains("black"))
        piece.classList.add("dragging")
        this.dragPiece.classList.add("show")
    }

    onDropPiece(event) {
        if (this.draggedPiece) {
            this.draggedPiece.classList.remove("dragging")
            this.draggedPiece = null
            this.dragPiece.classList.remove("show")
        }
        let tile = event.target
        if (!tile.classList.contains("tile")) {
            this.moveStart = null
            return
        }
        
        const start = this.moveStart
        if (start === null) {
            return
        }

        this.moveStart = null
        const idx = Array.from(tile.parentElement.childNodes).indexOf(tile)
        const x = idx % 8
        const y = Math.floor(idx / 8)

        this.tryMove(start, [x, y])
    }

    tryMove(from, to, promotion=null) {
        const [x1, y1] = from
        const [x2, y2] = to

        const pos1 = this.toAlgebraic(from)
        const pos2 = this.toAlgebraic(to, promotion)

        if (pos1 === pos2) {
            return
        }

        const fen = this.getFEN()
        
        if (this.state[y1][x1].toLowerCase() === Piece.Pawn && !promotion) {
            const flipped = this.currentPlayer !== this.pov
            if (y2 === (flipped ? 7 : 0)) {
                this.promptPromotion(from, to)
                return
            }
        }

        // Pre-move
        this.movePiece(x1, y1, x2, y2)
        const move = pos1 + pos2

        apiPost("/move/", {
            fen: fen,
            move: move
        }).then(res => {
            if (res.valid) {
                this.loadFEN(res.fen)
                this.onMoved(move, res.outcome)
            } else {
                // Cancel move
                this.loadFEN(fen)
            }
        })
    }

    updateDragPiece(event) {
        if (this.draggedPiece) {
            this.dragPiece.style.setProperty("--x", event.clientX + "px")
            this.dragPiece.style.setProperty("--y", event.clientY + "px")
        }
    }

    flip() {
        for (let y1=0; y1<4; y1++) {
            for (let x1=0; x1<8; x1++) {
                this.swapPieces(x1, y1, 7 - x1, 7 - y1)
            }
        }
    }

    movePiece(x1, y1, x2, y2) {
        this.swapPieces(x1, y1, x2, y2, true)
    }

    swapPieces(x1, y1, x2, y2, move=false) {
        const piece1 = this.state[y1][x1]
        const piece2 = this.state[y2][x2]

        this.state[y2][x2] = piece1
        this.state[y1][x1] = move ? "" : piece2

        const pieceNode1 = this.pieces[y1][x1]
        const pieceNode2 = this.pieces[y2][x2]

        const tile1 = this.board.childNodes[y1 * 8 + x1]
        const tile2 = this.board.childNodes[y2 * 8 + x2]

        tile1.innerHTML = ""
        tile2.innerHTML = ""
        
        if (pieceNode1) tile2.appendChild(pieceNode1)
        this.pieces[y2][x2] = pieceNode1
        
        if (!move && pieceNode2) {
            tile1.appendChild(pieceNode2)
        }
        this.pieces[y1][x1] = move ? null : pieceNode2
    }

    nextPlayer() {
        if (this.currentPlayer === "black") {
            this.moves++
        }
        this.currentPlayer = this.currentPlayer === "white" ? "black" : "white"
    }

    getFEN() {
        const pov = this.pov
        let fen = ""
        const state = pov === "black" ? this.getFlippedState() : this.state
        for (let y=0; y<8; y++) {
            if (y !== 0) {
                fen += "/"
            }
            let empty = 0
            for (let x=0; x<8; x++) {
                const piece = state[y][x]
                if (piece) {
                    if (empty) {
                        fen += empty
                        empty = 0
                    }
                    fen += piece
                } else {
                    empty++
                }
            }
            if (empty) {
                fen += empty
            }
        }
        fen += " " + this.currentPlayer.slice(0, 1)
        let castles = ""
        if (this.whites.canSmallCastle) castles += "K"
        if (this.whites.canBigCastle) castles += "Q"
        if (this.blacks.canSmallCastle) castles += "k"
        if (this.blacks.canBigCastle) castles += "q"
        if (!castles) castles = "-"
        fen += " " + castles
        fen += " " + (this.enPassant ? this.toAlgebraic(this.enPassant) : "-")
        fen += " " + this.halfMoves
        fen += " " + this.moves

        return fen
    }

    getFlippedState() {
        return this.state.map(r => r.toReversed()).toReversed()
    }

    toAlgebraic(pos, promotion=null) {
        let [x, y] = pos
        if (this.pov === "black") {
            x = 7 - x
            y = 7 - y
        }
        let rank = 8 - y
        let file = Chessboard.FILES.at(x)
        return file + rank + (promotion ?? "")
    }

    fromAlgebraic(pos) {
        let x = Chessboard.FILES.indexOf(pos.at(0))
        let y = 8 - (+pos.at(1))
        if (this.pov === "black") {
            x = 7 - x
            y = 7 - y
        }
        return [x, y]
    }

    promptPromotion(from, to) {
        this.promotionMove = [from, to]
        this.promotionPopup.show()
    }

    confirmPromotion(btn) {
        if (this.promotionMove) {
            const [from, to] = this.promotionMove
            this.promotionMove = null
            this.tryMove(from, to, btn.dataset.piece)
        }
        this.promotionPopup.hide()
    }

    onMoved(algebraic, outcome) {
        this.history.push(algebraic)
        if (outcome) {
            this.onGameOver(outcome)
            return
        }

        if (this.currentPlayer !== this.pov) {
            this.evaluate()
            this.predict()
        }
    }

    evaluate() {
        apiPost("/evaluate/", {
            "player": this.currentPlayer,
            "moves": this.history
        }).then(res => {
            document.getElementById("estimated-elo").innerText = res.elo
        })
    }

    predict() {
        apiPost("/predict/", {
            "fen": this.getFEN(),
            "elo": 850
        }).then(res => {
            if (res.error) {
                window.alert(res.error)
                return
            }

            this.loadFEN(res.fen)
            this.onMoved(res.move, res.outcome)
        })
    }

    onGameOver(outcome) {
        this.emit(new GameOverEvent(outcome))
    }
}

/**
 * @callback ChessListener
 * @param {ChessEvent}
 */

class Player {
    constructor(color) {
        this.color = color
        this.canBigCastle = true
        this.canSmallCastle = true
    }
}

class Game {
    constructor() {
        this.board = new Chessboard(document.getElementById("chessboard"))
        this.startBtn = document.getElementById("start")
        this.gameOverPopup = new Popup(document.getElementById("game-over-popup"))

        this.initListeners()
    }

    initListeners() {
        this.startBtn.addEventListener("click", () => this.start())
        this.board.addListener("gameOver", e => this.onGameOver(e))
        this.gameOverPopup.node.querySelector("button").addEventListener("click", () => this.gameOverPopup.hide())
    }

    start() {
        let playAs = document.getElementById("play-as").value
        if (playAs === "random") {
            playAs = Math.random() < 0.5 ? "white" : "black"
        }
        this.board.setPOV(playAs)
        this.board.loadFEN(Chessboard.DEFAULT_STATE)
    }

    onGameOver(event) {
        /** @type {GameOutcome} */
        const outcome = event.outcome
        const popup = this.gameOverPopup

        let termination
        let result = outcome.result.split("-").join(" - ")
        let winner = ""

        switch (outcome.termination) {
            case "CHECKMATE":
                termination = "Checkmate"
                winner = outcome.winner.slice(0, 1).toUpperCase() + outcome.winner.slice(1).toLowerCase() + "s won"
                break
            
            case "STALEMATE":
                termination = "Stalemate"
                break
            
            case "INSUFFICIENT_MATERIAL":
                termination = "Insufficient material"
                break
        
            default:
                termination = "Game Over"
                break
        }

        popup.node.querySelector(".termination").innerText = termination
        popup.node.querySelector(".result").innerText = result
        popup.node.querySelector(".winner").innerText = winner
        popup.show()
    }
}


window.addEventListener("load", () => {
    window.game = new Game()
})