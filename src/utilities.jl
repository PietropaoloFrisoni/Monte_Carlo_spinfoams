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

# contract a vertex tensor with the 6j matrix (right) and vector with phases (bottom right)
function tensor_contraction!(tensor_with_phase, original_tensor, W6j_matrix, vec_with_phases)

  @turbo for i1 in axes(tensor_with_phase, 1), i2 in axes(tensor_with_phase, 2), i3 in axes(tensor_with_phase, 3), i4 in axes(tensor_with_phase, 4), i5 in axes(tensor_with_phase, 5)

    for k_r in axes(W6j_matrix, 1)
      tensor_with_phase[i1, i2, i3, i4, i5] = original_tensor[i1, i2, i3, i4, k_r] * W6j_matrix[k_r, i5]
    end

    tensor_with_phase[i1, i2, i3, i4, i5] *= vec_with_phases[i4]

  end
end
