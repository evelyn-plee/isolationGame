import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def heuristic0(game, player):
    pass

def heuristic1(game, player):
    pass

def heuristic2(game, player):
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    occupied_factor = 1
    if len(game.get_blank_spaces() < game.width * game.height/4):
        occupied_factor = 4

    corners = [(0,0), ((game.height-1),0), (0,(game.width-1)), ((game.height-1), (game.weight-1))]

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    corners_for_own = [move for move in own_moves if move in corners]
    corners_for_opp = [move for move in opp_moves if move in corners]

    own_weightage = len(own_moves)-(occupied_factor*len(corners_for_own))
    opp_weightage = len(opp_moves)-(occupied_factor*len(corners_for_opp))

    return float(own_weightage - opp_weightage)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        move = (-1, -1)
        if not legal_moves:
            return (-1, -1)

        depth = 1 if self.iterative else self.search_depth

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                while True:
                    if self.method == 'minimax':
                        _, move = self.minimax(game, depth)
                    elif self.method == 'alphabeta':
                        _, move = self.alphabeta(game, depth)
                    else:
                        raise NotImplementedError
                depth += 1
            else:
                if self.method == 'minimax':
                    _, move = self.minimax(game, depth)
                elif self.method == 'alphabeta':
                    _, move = self.alphabeta(game, depth)
                else:
                    raise NotImplementedError
                return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get legal moves for active player
        legal_moves = game.get_legal_moves()
        # Check if there isn't anymore legal moves, then return
        # -inf or +inf depending on "maximizing_player"
        if not legal_moves:
            return game.utility(self), (-1,-1)

        if depth <= 0:
            return self.score(game, self), (-1,-1)

        best_move = None
        if maximizing_player:
            best_score = float("-inf")
            for move in legal_moves:
                advance_state = game.forecast_move(move)
                score, _ = self.minimax(advance_state, depth-1, False)
                if score > best_score:
                    best_score, best_move = score, move
        else:
            best_score = float("inf")
            for move in legal_moves:
                advance_state = game.forecast_move(move)
                score, _ = self.minimax(advance_state, depth-1, True)
                if score < best_score:
                    best_score, best_move = score, move
        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self), (-1,-1)

        if depth <= 0:
            return self.score(game, self), (-1, -1)

        best_move = None
        if maximizing_player:
            best_score = float("-inf")
            for move in legal_moves:
                advance_state = game.forecast_move(move)
                score, _ = self.alphabeta(advance_state, depth-1, alpha, beta, False)
                alpha = max(alpha, score)
                if score > best_score:
                    best_score, best_move = score, move
                if alpha >= beta:
                    break
        else:
            best_score = float("inf")
            for move in legal_moves:
                advance_state = game.forecast_move(move)
                score, _ = self.alphabeta(advance_state, depth-1, alpha, beta, True)
                beta = min(beta, score)
                if score < best_score:
                    best_score, best_move = score, move
                if alpha >= beta:
                    break

        return best_score, best_move
