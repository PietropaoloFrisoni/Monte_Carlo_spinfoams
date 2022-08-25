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