import torch
import pytest


def test_matmul_basic():
    """Test basic matrix multiplication with known values."""
    # Create two simple matrices
    # A = [[1, 2],
    #      [3, 4]]
    A = torch.tensor([[1.0, 2.0, 3.0],
                      [3.0, 4.0, 2.0]])
    
    print(A.shape)

    # B = [[5, 6],
    #      [7, 8]]
    B = torch.tensor([[5.0],
                      [7.0],
                      [4.0],])
    
    print(B.shape)

    expected = torch.tensor([[31.0, 51.0]]).T
    
    result = A @ B
    print(result.shape)
    print(result)

    print(expected.shape)
    print(expected)

    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
    assert result.shape == (2, 1), f"Expected shape (2, 1), but got {result.shape}"


def test_3_dims():
    A = torch.tensor([
        [[1, 2, 3, 4],
        [3, 2, 4, 2],
        [3, 2, 4, 2]],
        [[4, 1, 1, 3],
        [1, 4, 2, 1],
        [1, 4, 2, 1]]
    ])

    B = torch.tensor([
        [1, 2, 3, 5],
        [3, 2, 3, 5],
        [1, 3, 3, 5]
    ])

    print(A.shape)
    print(B.shape)

    C = torch.matmul(A, B.T)
    print(C.shape)
    print(C)

    assert C.shape == (2, 3, 3)

def test_understand_transpose():
    A = torch.tensor([
        [[1,2,3], 
         [4,5,6]],
        [[11,12,13], 
         [14,15,16]],
        [[111,112,113], 
         [114,115,116]],
        [[1111,1112,1113], 
         [1114,1115,1116]],
    ])

    print(A)
    print(A.shape)

    B = A.transpose(0, 1)
    print(B)
    print(B.shape)

    C = A.transpose(0, 2)
    print(C)
    print(C.shape)

    D = A.transpose(1, 2)
    print(D)
    print(D.shape)

    assert D.shape == (4, 3, 2)
    