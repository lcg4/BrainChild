#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main()
{
    printf("hey luke, the tensorflow lib is: %s\n", TF_Version());
}
