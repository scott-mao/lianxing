
#选择排序方式
def seclection_sort(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i,n):
            if nums[i] > nums[j]:
                nums[i],nums[j] = nums[j],nums[i]
    return nums

def bubble_sort(nums):
    n= len(nums)
    for i in range(n):
        for j in range(1,n-i):
            if nums[j-1] > nums[j]:
                nums[j-1],nums[j] = nums[j],nums[j-1]
    return nums

def insertion_sort(nums):
    n = len(nums)
    for i in range(1,n):
        while i>0 and nums[i] < nums[i-1]:
            nums[i],nums[i-1] = nums[i-1],nums[i]
            i -= 1
    return nums
def shell_sort(nums):
    n = len(nums)
    gap = n//2
    while gap:
        for i in range(gap,n):
            while (i-gap >= 0) and nums[i-gap] > nums[i]:
                nums[i-gap],nums[i] = nums[i],nums[i-gap]
                i = i-gap
        gap //= 2
    return nums


def merge_sort(nums):
    n = len(nums)
    if n == 1:
        return nums
    mid = n//2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left,right)
def merge(left,right):
    i = 0
    j = 0
    res = []
    if i<=len(left) and j<=len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res += left[i:]
    res += right[j:]
    return res



def quick_sort(nums):
    n = len(nums)
    def quick(left, right):
        if left >= right:
            return nums
        pivot = left
        i = left
        j = right
        while i < j:
            while i < j and nums[j] > nums[pivot]:
                j -= 1
            while i < j and nums[i] <= nums[pivot]:
                i += 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i], nums[pivot] = nums[pivot], nums[i]
        quick(left, i - 1)
        quick(i + 1, right)
        return nums
    return quick(0,n-1)





# def merge_sort(nums):
#     if len(nums) <= 1:
#         return nums
#     mid = len(nums) // 2
#     # 分
#     left = merge_sort(nums[:mid])
#     right = merge_sort(nums[mid:])
#     # 合并
#     return merge(left, right)
#
#
# def merge(left, right):
#     res = []
#     i = 0
#     j = 0
#     while i < len(left) and j < len(right):
#         if left[i] <= right[j]:
#             res.append(left[i])
#             i += 1
#         else:
#             res.append(right[j])
#             j += 1
#     res += left[i:]
#     res += right[j:]
#     return res





if __name__ == '__main__':
    nums = [1,4,5,3,1,2,0]
    print("原列表中的元素"+str(nums))


    #选择排序
    flag = 0
    if flag == 1:
        seclec_nums =seclection_sort(nums)
        print("选择排序结果"+str(seclec_nums))

    #冒泡排序
    flag = 0
    if flag==1:
        bubble_nums = bubble_sort(nums)
        print("冒泡排序结果"+str(bubble_nums))


    #插入排序
    flag = 0
    if flag == 1:
        insertion_nums = insertion_sort(nums)
        print("插入排序结果"+str(insertion_nums))

    #希尔排序
    flag = 0
    if flag ==1:
        shell_nums = shell_sort(nums)
        print("希尔排序结果"+str(shell_nums))

    #归并排序
    flag = 0
    if flag ==1:
        merge_nums=merge_sort(nums)
        print("归并排序结果" + str(merge_nums))

    #快速排序
    flag = 1
    if flag == 1:
        quick_nums = quick_sort(nums)
        print("快速排序结果" + str(quick_nums))

