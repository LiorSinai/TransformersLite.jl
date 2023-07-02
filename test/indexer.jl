using TransformersLite: IndexTokenizer
using TransformersLite: decode

@testset "Indexer" begin
    ## letters
    indexer = IndexTokenizer(["aa","bb","cc"], "[UNK]")
    indices = indexer(["cc", "aa", "aa", "dd", "bb"])
    expected_indices = [4, 2, 2, 1, 3]
    @test indices == expected_indices

    letters = decode(indexer, [4, 2, 2, 1, 3])
    expected_letters = ["cc", "aa", "aa", "[UNK]", "bb"]
    @test letters == expected_letters

    indices = indexer([["cc", "aa"], ["aa", "dd", "bb"], ["bb"]])
    expected_indices = hcat([4, 2, 1], [2, 1, 3], [3, 1, 1])
    @test indices == expected_indices

    indices = hcat([4, 2, 1], [2, 1, 3], [3, 1, 1])
    letters = decode(indexer, indices)
    expected_letters = [["cc", "aa", "[UNK]"], ["aa", "[UNK]", "bb"], ["bb", "[UNK]", "[UNK]"]]
    @test letters == expected_letters

    ## words
    indexer = IndexTokenizer(["this","book","recommend","highly"], "[UNK]")
    indices = indexer(["i", "highly", "recommend", "this", "book", "by", "brandon", "sanderson"])
    expected_indices = [1, 5, 4, 2, 3, 1, 1, 1]
    @test indices == expected_indices

    words = decode(indexer, [1, 5, 4, 2, 3, 1, 1, 1])
    expected_words = ["[UNK]", "highly", "recommend", "this", "book", "[UNK]", "[UNK]", "[UNK]"]
    @test words == expected_words
end