; Declare the external 'add' function (it exists in our other file)
declare i32 @add(i32, i32)

define i32 @main() {
  ; Call 'add' with arguments 10 and 32
  %result = call i32 @add(i32 10, i32 32)
  
  ; Return the result as the exit code
  ret i32 %result
}