#####
## Layer and Network Architectures for TintiNet.jl
#

# 1. 1D Inception V2 layer with 3 window sizes (Inception4)

"""
Structure for the 1D Inception V2 layer with 3 window sizes (singleton, trigram, pentagram) and a pooling layer
"""
struct Inception4{L1, L2, L3, L4}
    Solo_One_Layer::L1
    One_Three_Layer::L2
    One_Five_Layer::L3
    Pool_One_Layer::L4
end

@functor Inception4

function Inception4(input_channels)
    output_channels = fld(input_channels, 4)
    return Inception4(
        Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 3), output_channels => output_channels, relu; pad = (0,1))
        ),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 5), output_channels => output_channels, relu; pad = (0,2))
        ),
        Chain(
            MaxPool((1,3); pad=(0,1), stride=1),
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0))
        ),
    )
end

function (m::Inception4)(x)

    res1 = m.Solo_One_Layer(x)
    res2 = m.One_Three_Layer(x)
    res3 = m.One_Five_Layer(x)
    res4 = m.Pool_One_Layer(x)

    concatenated_result = cat(res1, res2, res3, res4; dims = (3))

end

# 2. 1D Inception V2 layer with 7 window sizes (Inception8 or Inceptigor)

"""
Structure for the 1D Inception V2 layer with 7 window sizes and a pooling layer
"""
struct Inception8{L1, L2, L3, L4, L5, L6, L7, L8}
    Solo_One_Layer::L1
    One_Three_Layer::L2
    One_Five_Layer::L3
    One_Seven_Layer::L4
    One_Nine_Layer::L5
    One_Eleven_Layer::L6
    One_Thirteen_Layer::L7
    Pool_One_Layer::L8
end

@functor Inception8

function Inception8(input_channels)
    output_channels = fld(input_channels, 8)
    return Inception8(
        Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 3), output_channels => output_channels, relu; pad = (0,1))
        ),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 5), output_channels => output_channels, relu; pad = (0,2))
        ),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 7), output_channels => output_channels, relu; pad = (0,3))
        ),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 9), output_channels => output_channels, relu; pad = (0,4))
        ),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 11), output_channels => output_channels, relu; pad = (0,5))
        ),
        Chain(
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0)),
            Conv((1, 13), output_channels => output_channels, relu; pad = (0,6))
        ),
        Chain(
            MaxPool((1,3); pad=(0,1), stride=1),
            Conv((1, 1), input_channels => output_channels, relu; pad = (0,0))
        ),
    )
end

function (m::Inception8)(x)

    res1 = m.Solo_One_Layer(x)
    res2 = m.One_Three_Layer(x)
    res3 = m.One_Five_Layer(x)
    res4 = m.One_Seven_Layer(x)
    res5 = m.One_Nine_Layer(x)
    res6 = m.One_Eleven_Layer(x)
    res7 = m.One_Thirteen_Layer(x)
    res8 = m.Pool_One_Layer(x)

    concatenated_result = cat(res1, res2, res3, res4, res5, res6, res7, res8; dims = (3))

end
