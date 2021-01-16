import json
import re
import argparse
import numpy as np
import pickle
from typing import List, Tuple, Union

SPACE = 0
BLACK = 1 
WHITE = 2
SIZE = 19

def shape(ndarray: Union[List, float]) -> Tuple[int, ...]:
    if isinstance(ndarray, list):
        # More dimensions, so make a recursive call
        outermost_size = len(ndarray)
        row_shape = shape(ndarray[0])
        return (outermost_size, *row_shape)
    else:
        # No more dimensions, so we're done
        return ()

def OPPNUM(colornum):
    return colornum%2 + 1

def charchar2pos(s):
    _x = ord(s[0])-97
    _y = ord(s[1])-97
    return _y*19+_x

def pos2charchar(pos):
    _x = pos%19
    _y = int(pos/19)
    
    a = chr(_x+97)
    b = chr(_y+97)
    return a+b


def err(msg):
    print('error:', msg)

class MOVE:
    def __init__(self, color, pos, comment = 'None'):
        self.color = color
        self.pos = pos
        self.comment = comment    

    def GetColor(self):
        return 'B' if self.color == 'B' else 'W'
    
    def GetPositionStr(self):
        alphabets = 'ABCDEFGHJKLMNOPQRSTU'
        return alphabets[ int(self.pos/19) ] + str(self.pos%19 + 1)
    
    def SetComment(self, comment):
        self.comment = comment
    
    def __str__(self):
        return ';{}[{}]'.format(self.color, self.pos)

class BOARD:
    def __init__(self):
        self.FileName = "None"
        self.board = [SPACE] * SIZE * SIZE

    def PutStone(self, move):
        pos = move.pos
        #global current_sgfstr
        if(self.board[pos]):
            err('Stone exists.' + move.__str__())
        else:
            pass
        self.board[pos] = move.color
        legal_nearby_pos = self.GetLegalNearby(pos)
        for newpos in legal_nearby_pos:
            if self.board[newpos] == OPPNUM(self.board[pos]):
                chi,visited = self.GetBlock(newpos , chi=set(), visited=set())#以pos位置為中心，每次跑完要重設set()
                if len(chi) == 0:
                    self.ClearStone(visited)
    
    def ClearStone(self, death_stones):
        for deadstone_pos in death_stones:
            self.board[deadstone_pos] = 0        
    
    def GetLegalNearby(self, pos):
        x = pos%19
        y = int(pos/19)
        left = [-1,0]
        right = [1,0]
        up = [0, -1]
        down = [0, 1]
        directions = [left, right, up , down]
        if x == 0:
            directions.remove(left)
        if x == 18:
            directions.remove(right)
        if y == 0:
            directions.remove(up)
        if y == 18:
            directions.remove(down)        

        legal_nearby_pos = []
        
        for direction in directions:
            _x = x + direction[0]
            _y = y + direction[1]
            legal_nearby_pos.append( 19*_y + _x )
            
        return legal_nearby_pos

    def GetBlock(self, pos, chi, visited):
        visited.add(pos)
        if self.board[pos] == SPACE:
            return set(),set()
        legal_nearby_pos = self.GetLegalNearby(pos)
        connected = []
        for newpos in legal_nearby_pos:
            if newpos in visited:
                continue
            if self.board[ newpos ] == self.board[pos]:
                connected.append(newpos)
            elif self.board[newpos] == SPACE:
                chi.add(newpos)

        for _pos in connected:
            self.GetBlock(_pos, chi, visited)
            
        return chi, visited
    
    def PrintBoard(self):

        for pos in range(len(self.board)):
            x = pos%19
            y = int(pos/19)
            c = ''

            if self.board[pos] == 1:
                c = 'X'
            elif self.board[pos] == 2:
                c = 'O'
            else:
                c = '.'
            print( c , end=' ')

            if x == 18:
                print()
        print()
   
    def __eq__(self, other):
        return other.board == self.board
    
    def __add__(self, move):
        newb = BOARD()
        newb.board = self.board.copy()
        newb.PutStone(move)
        return newb

class GAME:
    def __init__(self):
        self.FileName = '(NULL)'
        self.boards = []
        self.ch_keys = []
        self.Moves = []

    def Parser(self, sgf_str):
        def make_tag_idx_pairs(tags, idx, tagname):
            if len(tags) != len(idx):
                print('ERROR')           
            return [[tagname, idx[i], tags[i]] for i in range(len(idx))]
        
        self.sgf_formats = []
        
        self.sgf_str = sgf_str[:sgf_str.find(')')+1]
        TagBPattern = '[^AP]B(\[..\])+'
        TagWPattern = '[^AP]W(\[..\])+'
        self.BTag = [ x[1:3].lower() for x in re.findall(TagBPattern, self.sgf_str)]
        self.BIdx = [ m.start(0) for m in re.finditer(TagBPattern, self.sgf_str)]
        self.WTag = [ x[1:3].lower() for x in re.findall(TagWPattern, self.sgf_str)]
        self.WIdx = [ m.start(0) for m in re.finditer(TagWPattern, self.sgf_str)]        
    
        self.chrono = []
        for Q in [(self.BTag, self.BIdx, 'B'), (self.WTag, self.WIdx, 'W')]:
            self.chrono.extend(make_tag_idx_pairs(Q[0], Q[1], Q[2])) # tagname , char_idx, tags_content
        self.chrono = sorted(self.chrono, key=lambda x: x[1])

        for mv in self.chrono:
            # if the move is pass, then drop it.
            if 'tt' in mv[2]:
                continue
            color = BLACK if mv[0] == 'B' else WHITE#black1white2
            pos = charchar2pos(mv[2])#0~360
            self.Moves.append(MOVE(color, pos))

    def GetComment(self, sgf_str):
        comment = sgf_str.split('|')
        comment = comment[-1].replace(' ', '').replace('\n', '').replace('(', '').replace(')','')
        comment = comment.replace('[', '').replace(']', '').replace("'", '').replace(',', ';')
        comment = comment.replace('W', '').replace('B', '').replace('C', '')
        comment = comment.replace('a', '').replace('b', '').replace('c', '').replace('d', '')
        comment = comment.replace('e', '').replace('f', '').replace('g', '').replace('h', '')
        comment = comment.replace('i', '').replace('j', '').replace('k', '').replace('l', '')
        comment = comment.replace('m', '').replace('n', '').replace('o', '').replace('p', '')
        comment = comment.replace('q', '').replace('r', '').replace('s', '').replace('t', '')
        
        return comment


    def GetMoveNum(self):
        return len(self.Moves)

    def GetMove(self, MoveNum):
        return self.Moves[MoveNum]

    def PlayTo(self, MoveNum):
        self.boards = []
        cur = BOARD()
        for i in range(0, min(MoveNum, self.GetMoveNum())):
            cur += self.GetMove(i)
            self.boards.append(cur)
            #cur.PrintBoard()
            
    
    def SaveTermFeature(self,BW_feature,onehot_comment_label, comment, pos_num):
        self.PlayTo(500)
        Move_num = self.GetMoveNum()
        BW_feature.append(board2numpyfeature(self.boards[Move_num-1].board, self.boards[Move_num-2].board, self.Moves[Move_num-1].color))
        onehot_comment_label.append(self.comment2onehot(comment, pos_num))


    def comment2onehot(self, comment, pos_num):
        onehot = [0]*len(self.ch_keys)
        for _comment in comment.split(';'):
            try:
                idx = self.ch_keys.index(_comment)
                onehot[idx] = 1
            
            except:
                try:
                    stop = 0
                    for i in range(len(_comment)+1):
                        if _comment[stop:i] in self.ch_keys:
                            idx = self.ch_keys.index(_comment[stop:i])
                            onehot[idx] = 1
                            stop = i
                    
                except:
                    stop = 0
                    flag = 0
                    for i in range(len(_comment)+1):
                        if _comment[stop:i] in self.ch_keys:
                            if flag == 1:
                                idx = self.ch_keys.index(_comment[stop:i])
                                onehot[idx] = 1
                                stop = i
                                flag = 0
                            else:
                                flag = 1
                
                if _comment == '':
                    continue
                
                #if 'None' not in _comment:
                #    print(pos_num)
                #    err(_comment + ' not in lib')
                    
        return np.array(onehot)


def LoadLibrary():
    ch_keys = []
    with open('./library.json', 'r',encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            Q = json.loads(line)
            if Q['chinese_name'] not in ch_keys:
                ch_keys.append( Q['chinese_name'] )
    ch_keys.append('None')

    return ch_keys
    
def board2numpyfeature(board, past_board, color):
    B, W = [], []
    B_past, W_past = [], []
    feature = np.zeros((4, 19, 19), dtype = np.int)
    
    for i in range(len(board)):
        if board[i] == BLACK:
            B.append(1)
            W.append(SPACE)            
        elif board[i] == WHITE:
            B.append(SPACE)
            W.append(1)
        else:
            B.append(0)
            W.append(0)

    for i in range(len(past_board)):
        if past_board[i] == BLACK:
            B_past.append(1)
            W_past.append(SPACE)            
        elif past_board[i] == WHITE:
            B_past.append(SPACE)
            W_past.append(1)
        else:
            B_past.append(0)
            W_past.append(0)
    
    B, W = np.array(B), np.array(W)
    B_past, W_past = np.array(B_past), np.array(W_past)

    if color == BLACK:
        feature[0], feature[1] = np.reshape(B, (19, 19)), np.reshape(W, (19, 19))
        feature[2], feature[3] = np.reshape(B_past, (19, 19)), np.reshape(W_past, (19, 19))
    else:
        feature[0], feature[1] = np.reshape(W, (19, 19)), np.reshape(B, (19, 19))
        feature[2], feature[3] = np.reshape(W_past, (19, 19)), np.reshape(B_past, (19, 19))
    
    return feature
    
def board2board(vboard):
    board = []
    for ch in vboard.replace('\n', '').replace(' ', ''):
        board.append(int(ch))
    return board

def sgfs2features(args, FileName):
    ch_keys = LoadLibrary()
    b = BOARD()
    with open(FileName, 'r' , encoding="utf-8", errors='ignore') as fin:
        BW_feature = []
        onehot_comment_label = []
        pos_num = 0
        for line in fin.readlines():
            g = GAME()
            g.ch_keys = ch_keys
            g.Parser(line)
            comment = g.GetComment(line)
            #print(comment)
            g.SaveTermFeature(BW_feature, onehot_comment_label, comment, pos_num+1)
            pos_num += 1
            if pos_num % 1000 == 0:
                print("Loading {pos_num} Positions...".format(pos_num = pos_num))
    
    return BW_feature, onehot_comment_label, len(ch_keys), ch_keys
