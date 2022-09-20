# returns number of files in a folder
function file_count(folder::String)
  files_and_dirs = readdir(folder)
  number_of_files = size(files_and_dirs)[1]
  return number_of_files
end

# logging function (flushing needed)
function log(x...)
  println("[ ", now(), " ] - ", join(x, " ")...)
  flush(stdout)
end

# comunicate between processes
macro retrieve_from_process(p, obj, mod=:Main)
  quote
    remotecall_fetch($(esc(p)), $(esc(mod)), $(QuoteNode(obj))) do m, o
      Core.eval(m, o)
    end
  end
end

# check size of tensor
@inline function check_size(tensor_pre_contracted, original_tensor)
  if (size(tensor_pre_contracted) != size(original_tensor))
    error("\nThe 2 tensors have different sizes")
  end
end

# contract a vertex tensor with the 6j matrix (on the right)
function tensor_contraction!(tensor_pre_contracted, original_tensor, W6j_matrix)

  for i5 in axes(tensor_pre_contracted, 1), i4 in axes(tensor_pre_contracted, 2), i3 in axes(tensor_pre_contracted, 3), i2 in axes(tensor_pre_contracted, 4), i1 in axes(tensor_pre_contracted, 5)

    @turbo for k_r in axes(W6j_matrix, 1)
      tensor_pre_contracted[i5, i4, i3, i2, i1] += original_tensor[k_r, i4, i3, i2, i1] * W6j_matrix[k_r, i5]
    end

  end
end

# from index to intertwiner of tuple ((i_min, i_max), i_range)
@inline function from_index_to_intertwiner(tuple, index)
  return tuple[1][1] + index - 1
end
