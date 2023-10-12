For each model published in our paper, we upsampled each training sample by 5 times and combined with Alpaca data.

During training, only `input`, `output`, and `instruction` fields are used. `id` stores the corresponding WebQ question IDs, and `sparal` stores the executable SPARQL (before any conversion) on WikiData.