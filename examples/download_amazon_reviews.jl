#=
This script mimics download_amazon_reviews.py to show that a pure Julai implementation is possible.
The Python script is still the recommended way because:
    1. The data is hosted by HuggingFace and any changes to it will be propagated directly to their API.
    2. It is much easier to edit that script to use a different dataset.
    3. The datasets API has more advanced functionality like caching data transforms.
=#

using Downloads
using ProgressMeter
using JSON
using DataFrames
using Arrow
using Printf
using SHA

function humanize(x::Int) 
    suffixes = ["B", "kB", "MB", "GB", "TB"]
    value, suffix_biggest = x, suffixes[end]
    unit = 1
    for suffix in suffixes
        if x < unit * 1024
            suffix_biggest = suffix
            break
        end
        unit *= 1024
    end
    value = x / unit
    @sprintf "%.2f %s" value suffix_biggest
end

function update_progress!(progressMeter::Progress, total::Int, now::Int)
    if (total == 0.0) || (progressMeter.counter == 100)
        return
    end
    counter = floor(Int, now / total * 100) 
    ProgressMeter.update!(progressMeter, counter, desc=humanize(total))
end

########### --------------------- Config --------------------- ###########
## source: https://huggingface.co/datasets/amazon_reviews_multi
base_dir = "datasets"
language = "en"
info_url = "https://huggingface.co/datasets/amazon_reviews_multi/raw/main/dataset_infos.json"
data_urls = [
    "https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/$split/dataset_$(language)_$split.json" 
    for split in ["train", "test", "dev"]
]
transformed_names = [
    "amazon_reviews_multi-$split.arrow" for split in ["train", "test", "validation"]
]
fingerprint = "0" # The fingerprint is a 64 character (1024 byte) xxHash that HuggingFace uses to cache data
                  # transforms. A dummy value is sufficient for this codebase.
int_columns = [:stars]   # convert these columns to integers
float_columns = Symbol[] # convert these columns to floats

########### --------------------- Base directory --------------------- ###########

if !(isdir(base_dir))
    mkdir(base_dir)
end

########### --------------------- Download --------------------- ###########

downloads_path = joinpath(base_dir, "downloads")
if !(isdir(downloads_path))
    mkdir(downloads_path)
end

info_filename = basename(info_url)
info_path = joinpath(downloads_path, info_filename)
Downloads.download(info_url, info_path)
info = JSON.parsefile(info_path)
info_all = info["all_languages"]
info_lang = info[language]
checksums = info_lang["download_checksums"]

download_paths = [joinpath(downloads_path, basename(url)) for url in data_urls]
for (url, output_path) in zip(data_urls, download_paths)
    println("Downloading $url to $output_path")
    progressMeter = Progress(100)
    Downloads.download(url, output_path; progress=(total, now)->update_progress!(progressMeter, total, now))
    print("Checking checksum ...")
    checksum = checksums[url]["checksum"]
    hash = open(output_path) do f
        bytes2hex(sha2_256(f))
    end
    if (hash != checksum)
        throw("checksum invalid for $output_path")
    end
    println(" valid")
end

println("")

########### --------------------- Transform --------------------- ###########

transformed_dir = joinpath(
    base_dir, info_all["builder_name"], language, info_all["version"]["version_str"], fingerprint
    )
if !(isdir(transformed_dir))
    mkpath(transformed_dir)
end

info_out_path = joinpath(transformed_dir, "dataset_info.json")
string_data = JSON.json(info_lang)
println("writing metadata to $info_out_path")
open(info_out_path, "w") do f
    write(f, string_data)
 end

for (raw_path, transformed_name) in zip(download_paths, transformed_names)
    println("transforming $raw_path")
    data = Dict{String, Union{Missing, String}}[]
    open(raw_path, "r") do f
        while ! eof(f) 
            line = readline(f)
            entry = JSON.parse(line)
            push!(data, entry)
        end
    end
    df = DataFrame(data)
    for col in int_columns 
        df[!, col] = parse.(Int32, (df[!, col]))
    end
    for col in float_columns 
        df[!, col] = parse.(Float64, (df[!, col]))
    end
    output_path = joinpath(transformed_dir, transformed_name)
    println("writing to $output_path")
    Arrow.write(output_path, df)
end
