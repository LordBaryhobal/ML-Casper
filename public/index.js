const Piece = {
    King: "k",
    Queen: "q",
    Bishop: "b",
    Knight: "n",
    Rook: "r",
    Pawn: "p",

    isWhite: p => p.toUpperCase() === p,
    isBlack: p => p.toLowerCase() === p
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

class Chessboard {
    static DEFAULT_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    constructor(node) {
        /** @type {HTMLDivElement} */
        this.node = node
        this.board = this.node.querySelector(".board")
        this.state = make2DArray(8, 8, "")
        this.pieces = make2DArray(8, 8, null)
        this.moveStart = null
        this.draggedPiece = null
        this.dragPiece = this.node.querySelector("#dragged")
        this.init()
        this.renderPieces()
    }

    init() {
        this.loadFEN(Chessboard.DEFAULT_STATE)
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
    }

    /**
     * 
     * @param {string} fen 
     */
    loadFEN(fen) {
        let parts = fen.split(" ")
        const state = make2DArray(8, 8, "")
        let x = 0
        let y = 0
        for (let i=0; i<parts[0].length; i++) {
            const char = parts[0][i]
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
    }

    renderPieces() {
        this.pieces = make2DArray(8, 8, null)
        for (let y=0; y<8; y++) {
            for (let x=0; x<8; x++) {
                const piece = this.state[y][x]
                if (!piece) continue
                const kind = piece.toLowerCase()
                const isBlack = Piece.isBlack(piece)
                const elem = document.createElement("img")
                elem.src = `/images/${kind}.png`
                elem.classList.add("piece")
                elem.setAttribute("draggable", "true")
                if (isBlack) elem.classList.add("black")
                const tile = this.board.childNodes[y * 8 + x]
                tile.appendChild(elem)
                this.pieces[y][x] = elem
            }
        }
    }

    onPickupPiece(x, y, event) {
        const piece = this.pieces[y][x]
        if (!piece) return
        this.moveStart = [x, y]
        this.draggedPiece = piece
        this.updateDragPiece(event)
        piece.classList.add("dragging")
        this.dragPiece.src = piece.src
        this.dragPiece.classList.toggle("black", piece.classList.contains("black"))
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

    tryMove(from, to) {
        const [x1, y1] = from
        const [x2, y2] = to

        if (x1 === x2 && y1 === y2) {
            return
        }

        const piece1 = this.state[y1][x1]
        const piece2 = this.state[y2][x2]

        if (!piece1) {
            return
        }
        this.state[y2][x2] = piece1
        this.state[y1][x1] = ""

        const piece = this.pieces[y1][x1]
        const tile = this.board.childNodes[y2 * 8 + x2]
        tile.innerHTML = ""
        tile.appendChild(piece)
        this.pieces[y2][x2] = piece
        this.pieces[y1][x1] = null
    }

    updateDragPiece(event) {
        if (this.draggedPiece) {
            this.dragPiece.style.setProperty("--x", event.clientX + "px")
            this.dragPiece.style.setProperty("--y", event.clientY + "px")
        }
    }
}


window.addEventListener("load", () => {
    window.board = new Chessboard(document.getElementById("chessboard"))
})