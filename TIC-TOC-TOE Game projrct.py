import math

# Function to initialize the game board
def create_board():
    # Create a 3x3 board filled with empty spaces
    return [[' ' for _ in range(3)] for _ in range(3)]

# Function to print the current state of the board
def print_board(board):
    for row in board:
        print('|'.join(row))
        print('-' * 5)  # Row separator

# Check if there are moves left on the board
def moves_left(board):
    for row in board:
        if ' ' in row:
            return True
    return False

# Function to evaluate the score of the current board state
def evaluate(board):
    # Checking rows for victory
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != ' ':
            return 10 if board[row][0] == 'X' else -10

    # Checking columns for victory
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return 10 if board[0][col] == 'X' else -10

    # Checking diagonals for victory
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return 10 if board[0][0] == 'X' else -10
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return 10 if board[0][2] == 'X' else -10

    # If no winner, return 0 (draw)
    return 0

# Minimax algorithm to find the best move for the AI
def minimax(board, depth, is_maximizing):
    score = evaluate(board)

    # Return the score if someone has won or there are no more moves
    if score == 10 or score == -10:
        return score
    if not moves_left(board):
        return 0

    # AI's turn (maximizing)
    if is_maximizing:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = max(best, minimax(board, depth + 1, False))
                    board[i][j] = ' '  # Undo move
        return best

    # Human's turn (minimizing)
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = min(best, minimax(board, depth + 1, True))
                    board[i][j] = ' '  # Undo move
        return best

# Function to find the best move for the AI
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    # Evaluate all possible moves and pick the best one
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '  # Undo move

                # If the value of the current move is better than the best found so far, update best_move
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

# Function to check if there's a winner or if it's a draw
def check_winner(board):
    score = evaluate(board)
    if score == 10:
        return 'AI wins!'
    elif score == -10:
        return 'Human wins!'
    elif not moves_left(board):
        return 'It\'s a draw!'
    return None

# Main function to run the game
def play_game():
    board = create_board()
    human_turn = True  # Set human to go first (O)

    print("Welcome to Tic-Tac-Toe! You are 'O' and AI is 'X'.")
    print_board(board)

    while True:
        # Check if there's a winner after every turn
        result = check_winner(board)
        if result:
            print(result)
            break

        if human_turn:
            # Human's move
            try:
                row, col = map(int, input("Enter your move (row and column): ").split())
            except ValueError:
                print("Please enter valid numbers.")
                continue

            # Validate the move
            if board[row][col] != ' ':
                print("Invalid move! Cell already taken.")
            else:
                board[row][col] = 'O'
                human_turn = False
        else:
            # AI's move
            print("AI is making its move...")
            row, col = find_best_move(board)
            board[row][col] = 'X'
            human_turn = True

        # Print the board after each move
        print_board(board)

# Start the game
if __name__ == "__main__":
    play_game()
